import hyperspy.api as hs
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets, mkQApp
import numpy as np
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QMainWindow, QDockWidget, QMenuBar, QToolBar, \
    QStatusBar, QAction, QFileDialog, QHBoxLayout, QMdiArea, QMdiSubWindow, QLabel
from PyQt5.QtGui import QCursor, QFont, QPen, QBrush

font = {'color': 'b', 'font-size': '14pt'}
my_font = QFont("Times", 10, QFont.Bold)
my_font.setBold(True)


def set_plot_fonts(plot, color='w', font="Times", size=10, bold=False):
    for label in ['left', 'right', 'top', 'bottom']:
        my_font = QFont(font, size, QFont.Bold)
        plot.getAxis(label).setTickFont(my_font)
        plot.getAxis(label).setTextPen(color)


class EELS:
    def __init__(self, file):
        s = hs.load(file)
        self.haadf = s[0]
        self.eels_lowloss = s[1]
        self.eels_highloss = s[2]

    def get_spectral_resolution(self):
        return self.eels_lowloss.axes_manager["Energy loss"].scale

    def get_offset(self):
        return self.eels_highloss.axes_manager["Energy loss"].offset


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle('EELSpy')
        self.resize(1200, 1000)

        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)

        file_menu = self.menu_bar.addMenu("File")
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        self.tool_bar = QToolBar("Main Toolbar", self)
        self.addToolBar(self.tool_bar)
        self.tool_bar.addAction(open_action)
        self.tool_bar.addAction(exit_action)

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        self.mdi_area = QMdiArea()
        self.setCentralWidget(self.mdi_area)

        self.graphics_window = pg.GraphicsLayoutWidget()
        self.main_subwindow = QMdiSubWindow()
        self.main_subwindow.setWidget(self.graphics_window)
        self.mdi_area.addSubWindow(self.main_subwindow)
        self.main_subwindow.show()

        self.spectrum_subwindow = None
        self.spectrum_plot = None
        self.data = None
        self.selected_pixel_roi = None  # Store the selected pixel for highlighting

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "DM4 Files (*.dm4);;All Files (*)")
        if file_name:
            self.load_data(file_name)

    def load_data(self, file):
        self.data = EELS(file)
        summed_data = np.sum(self.data.eels_highloss.data, axis=2)
        min_val, max_val = np.min(summed_data), np.max(summed_data)
        rescaled_data = (summed_data - min_val) / (max_val - min_val) * 100
        item = self.create_matrix(np.transpose(rescaled_data))
        self.add_plot(item, name="mainw")
        self.status_bar.showMessage(f"Loaded file: {file}")

    def create_matrix(self, data):
        self.corrMatrix = data
        self.correlogram = pg.ImageItem()
        self.correlogram.setImage(self.corrMatrix)
        self.correlogram.mouseClickEvent = self.on_map_left_clicked
        return self.correlogram

    def add_plot(self, item, name=""):
        self.graphics_window.clear()
        self.plotItem = self.graphics_window.addPlot()
        self.plotItem.setLabel("left", "y dimension", **font)
        self.plotItem.setLabel("bottom", "x dimension", **font)
        self.plotItem.invertY(True)
        self.plotItem.setDefaultPadding(0.0)
        self.plotItem.addItem(item)
        self.plotItem.showAxes(True, showValues=(True, True, False, False), size=20)
        self.plotItem.getAxis('bottom').setHeight(10)

        colorMap = pg.colormap.get("CET-L1")
        bar = pg.ColorBarItem(colorMap=colorMap)
        bar.setImageItem(item, insert_in=self.plotItem)

    def on_map_left_clicked(self, event):
        if self.data is None:
            return
        if event.button() == 1 and event.modifiers() == Qt.ShiftModifier:
            pos = event.pos()
            x, y = int(pos.x()), int(pos.y())

            if self.selected_pixel_roi is None:
                # Create the ROI for the first time
                self.selected_pixel_roi = pg.RectROI([x - 0.5, y - 0.5], [1, 1], pen=pg.mkPen('r', width=2))
                self.plotItem.addItem(self.selected_pixel_roi)

                # Connect signal to update spectrum when ROI moves or resizes
                self.selected_pixel_roi.sigRegionChanged.connect(self.update_spectrum_from_roi)
            elif self.selected_pixel_roi.size()[0]>1 or self.selected_pixel_roi.size()[1]>1:
                self.selected_pixel_roi.setSize([1,1])
                self.selected_pixel_roi.setPos([x - 0.5, y - 0.5])
            else:
                # Just move the existing ROI
                self.selected_pixel_roi.setPos([x - 0.5, y - 0.5])

            # Immediately update spectrum for the new ROI position
            self.update_spectrum_from_roi()

    def update_spectrum_from_roi(self):
        """Extract all spectra inside ROI and update the spectrum plot."""
        if self.data is None or self.selected_pixel_roi is None:
            return

        # Extract the region of interest (ROI)
        roi_mask = self.selected_pixel_roi.getArrayRegion(self.data.eels_highloss.data, self.correlogram, axes=(1,0))

        if roi_mask is not None and roi_mask.size > 0:
            summed_spectrum = np.sum(roi_mask, axis=(0, 1))
        else:
            summed_spectrum = np.zeros_like(self.data.eels_highloss.data[0, 0])

        spectrum_res = self.data.get_spectral_resolution()
        spectrum_offset = self.data.get_offset()
        x_values = np.array(range(len(summed_spectrum))) * spectrum_res + spectrum_offset

        if self.spectrum_subwindow is None:
            self.spectrum_plot = SpectrumPlot()
            self.spectrum_subwindow = QMdiSubWindow()
            self.spectrum_subwindow.setWidget(self.spectrum_plot)
            self.mdi_area.addSubWindow(self.spectrum_subwindow)
            self.spectrum_subwindow.show()
        pt = self.selected_pixel_roi.pos()
        self.spectrum_plot.update_plot(x_values, summed_spectrum, pt.__reduce__()[1])


class SpectrumPlot(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.plot_widget = pg.PlotWidget(background='w')
        self.plot_widget.setLabel("left", "counts", **font)
        self.plot_widget.setLabel("bottom", "energy", units="keV", **font)
        set_plot_fonts(self.plot_widget, size=11, color='k')
        self.label = QLabel("", self)
        layout.addWidget(self.label)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)


    def update_plot(self, xaxis, energies, position):
        self.plot_widget.clear()
        self.plot_widget.plot(xaxis, energies, pen='k')
        self.label.setText(f"Pixel: {position}")


if __name__ == '__main__':
    mkQApp("Main")
    main_window = MainWindow()
    main_window.load_data("D:\\OneDrive - Uniwersytet Jagielloński\\Studia\\python\\EELSpy\\pythonProject1\\tests\\reference\\STEM SI_eels_O_Mn_La_do_Ca.dm4")
    main_window.show()
    pg.exec()
