import hyperspy.api as hs
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets, mkQApp
import numpy as np
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QMainWindow, QDockWidget, QMenuBar, QToolBar,
                             QStatusBar, QAction, QFileDialog, QHBoxLayout, QMdiArea, QMdiSubWindow, QLabel,
                             QPushButton, QGridLayout, QListWidget, QLineEdit)
from PyQt5.QtGui import QCursor, QFont, QPen, QBrush
from scipy import optimize

font = {'color': 'b', 'font-size': '14pt'}
my_font = QFont("Times", 10, QFont.Bold)
my_font.setBold(True)


def set_plot_fonts(plot, color='w', font="Times", size=10):
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
        super().__init__(*args, **kwargs)
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
            elif self.selected_pixel_roi.size()[0] > 1 or self.selected_pixel_roi.size()[1] > 1:
                self.selected_pixel_roi.setSize([1, 1])
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
        roi_mask = self.selected_pixel_roi.getArrayRegion(self.data.eels_highloss.data, self.correlogram, axes=(1, 0))

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


class LinearFitResult:
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def evaluate(self, x):
        return self.slope * x + self.intercept

    def __str__(self):
        return f"y = {self.slope:.6f}x + {self.intercept:.6f}"


class SpectrumPlot(QWidget):
    def __init__(self):
        super().__init__()

        # Main layout
        main_layout = QVBoxLayout()

        # Pixel position label
        self.label = QLabel("", self)
        main_layout.addWidget(self.label)

        # Plot widget
        self.plot_widget = pg.PlotWidget(background='w')
        self.plot_widget.setLabel("left", "counts", **font)
        self.plot_widget.setLabel("bottom", "energy", units="keV", **font)
        set_plot_fonts(self.plot_widget, size=11, color='k')
        main_layout.addWidget(self.plot_widget)

        # Controls layout
        controls_layout = QGridLayout()

        # Background fitting controls
        self.fit_bg_button = QPushButton("Fit Background")
        self.fit_bg_button.clicked.connect(self.fit_background)
        controls_layout.addWidget(self.fit_bg_button, 0, 0)

        self.reset_bg_button = QPushButton("Reset Background")
        self.reset_bg_button.clicked.connect(self.reset_background)
        controls_layout.addWidget(self.reset_bg_button, 0, 1)

        # Intersection control section
        controls_layout.addWidget(QLabel("Intersection Energy:"), 1, 0)
        self.intersection_value = QLineEdit()
        self.intersection_value.setReadOnly(True)
        controls_layout.addWidget(self.intersection_value, 1, 1)

        self.edge_name_input = QLineEdit()
        self.edge_name_input.setPlaceholderText("Edge name")
        controls_layout.addWidget(self.edge_name_input, 1, 2)

        self.save_edge_button = QPushButton("Save Edge")
        self.save_edge_button.clicked.connect(self.save_edge)
        controls_layout.addWidget(self.save_edge_button, 1, 3)

        # Saved edges list
        self.saved_edges_list = QListWidget()
        controls_layout.addWidget(QLabel("Saved Edges:"), 2, 0)
        controls_layout.addWidget(self.saved_edges_list, 3, 0, 1, 4)

        main_layout.addLayout(controls_layout)
        self.setLayout(main_layout)

        # Data storage
        self.xaxis = None
        self.spectrum = None
        self.bg_roi = None
        self.edge_roi = None
        self.bg_fit = None
        self.edge_fit = None
        self.bg_fit_line = None
        self.edge_fit_line = None
        self.intersection_point = None
        self.spectrum_curve = None
        self.saved_edges = {}

    def update_plot(self, xaxis, energies, position):
        if self.spectrum_curve is not None:
            self.plot_widget.removeItem(self.spectrum_curve)
        self.spectrum_curve = self.plot_widget.plot(xaxis, energies, pen='k')
        self.label.setText(f"Pixel: {position}")

        # Store the data
        self.xaxis = xaxis
        self.spectrum = energies

        # Create ROIs if they don't exist
        if self.bg_roi is None:
            # Create background ROI around 20% of the x-axis range
            x_range = xaxis[-1] - xaxis[0]
            roi_width = x_range * 0.2
            start_x = xaxis[0] + x_range * 0.1

            # Find y values within this x range
            mask = (xaxis >= start_x) & (xaxis <= start_x + roi_width)
            if np.any(mask):
                y_values = energies[mask]
                min_y = np.min(y_values)
                max_y = np.max(y_values)
                height = max_y - min_y

                # Create ROI
                self.bg_roi = pg.LinearRegionItem(
                    values=[start_x, start_x + roi_width],
                    brush=pg.mkBrush(128, 128, 255, 50),
                    pen=pg.mkPen('b', width=2),
                    movable=True
                )
                self.plot_widget.addItem(self.bg_roi)
                self.bg_roi.sigRegionChanged.connect(self.on_bg_roi_changed)

        # Create edge ROI if it doesn't exist
        if self.edge_roi is None:
            # Create edge ROI around 60% of the x-axis range
            x_range = xaxis[-1] - xaxis[0]
            roi_width = x_range * 0.2
            start_x = xaxis[0] + x_range * 0.6

            self.edge_roi = pg.LinearRegionItem(
                values=[start_x, start_x + roi_width],
                brush=pg.mkBrush(255, 128, 128, 50),
                pen=pg.mkPen('r', width=2),
                movable=True
            )
            self.plot_widget.addItem(self.edge_roi)
            self.edge_roi.sigRegionChanged.connect(self.on_edge_roi_changed)

        # Reset fits
        self.bg_fit = None
        self.edge_fit = None
        if self.bg_fit_line is not None:
            self.plot_widget.removeItem(self.bg_fit_line)
            self.bg_fit_line = None
        if self.edge_fit_line is not None:
            self.plot_widget.removeItem(self.edge_fit_line)
            self.edge_fit_line = None
        if self.intersection_point is not None:
            self.plot_widget.removeItem(self.intersection_point)
            self.intersection_point = None

        self.intersection_value.setText("")

    def fit_background(self):
        if self.xaxis is None or self.spectrum is None or self.bg_roi is None:
            return

        # Get the region boundaries
        min_x, max_x = self.bg_roi.getRegion()

        # Find the indices that fall within the region
        mask = (self.xaxis >= min_x) & (self.xaxis <= max_x)
        x_values = self.xaxis[mask]
        y_values = self.spectrum[mask]

        if len(x_values) < 2:  # Need at least 2 points for linear fit
            return

        # Fit a line to the data
        params, residuals, rank, sing, rcond = np.polyfit(x_values, y_values, 1, full=True)
        slope, intercept = params

        # Store the fit
        self.bg_fit = LinearFitResult(slope, intercept)

        # Update the background fit line
        if self.bg_fit_line is not None:
            self.plot_widget.removeItem(self.bg_fit_line)

        # Create a line that spans the entire x-axis
        x_min, x_max = self.xaxis[0], self.xaxis[-1]
        x_fit = np.array([x_min, x_max])
        y_fit = self.bg_fit.evaluate(x_fit)

        # Plot the fit line
        self.bg_fit_line = self.plot_widget.plot(x_fit, y_fit, pen=pg.mkPen('b', width=2, style=Qt.DashLine))

        # If we have both fits, calculate intersection
        if self.edge_fit is not None:
            self.calculate_intersection()

    def on_bg_roi_changed(self):
        # Background fit must be explicitly triggered by button press
        pass

    def on_edge_roi_changed(self):
        # Automatically update edge fit when ROI changes
        self.fit_edge()

    def fit_edge(self):
        if self.xaxis is None or self.spectrum is None or self.edge_roi is None:
            return

        # Get the region boundaries
        min_x, max_x = self.edge_roi.getRegion()

        # Find the indices that fall within the region
        mask = (self.xaxis >= min_x) & (self.xaxis <= max_x)
        x_values = self.xaxis[mask]
        y_values = self.spectrum[mask]

        if len(x_values) < 2:  # Need at least 2 points for linear fit
            return

        # Fit a line to the data
        params, residuals,rank, sing, rcond = np.polyfit(x_values, y_values, 1, full=True)
        slope, intercept = params

        # Store the fit
        self.edge_fit = LinearFitResult(slope, intercept)

        # Update the edge fit line
        if self.edge_fit_line is not None:
            self.plot_widget.removeItem(self.edge_fit_line)

        # Create a line that spans the entire x-axis
        x_min, x_max = self.xaxis[0], self.xaxis[-1]
        x_fit = np.array([x_min, x_max])
        y_fit = self.edge_fit.evaluate(x_fit)

        # Plot the fit line
        self.edge_fit_line = self.plot_widget.plot(x_fit, y_fit, pen=pg.mkPen('r', width=2, style=Qt.DashLine))

        # If we have both fits, calculate intersection
        if self.bg_fit is not None:
            self.calculate_intersection()

    def reset_background(self):
        if self.bg_fit_line is not None:
            self.plot_widget.removeItem(self.bg_fit_line)
            self.bg_fit_line = None

        self.bg_fit = None

        # Remove intersection point if it exists
        if self.intersection_point is not None:
            self.plot_widget.removeItem(self.intersection_point)
            self.intersection_point = None

        self.intersection_value.setText("")

    def calculate_intersection(self):
        if self.bg_fit is None or self.edge_fit is None:
            return

        # Solve for the intersection point: bg_slope*x + bg_intercept = edge_slope*x + edge_intercept
        # (bg_slope - edge_slope)*x = edge_intercept - bg_intercept
        # x = (edge_intercept - bg_intercept) / (bg_slope - edge_slope)

        if self.bg_fit.slope == self.edge_fit.slope:
            # Lines are parallel, no intersection
            return

        x_intersect = (self.edge_fit.intercept - self.bg_fit.intercept) / (self.bg_fit.slope - self.edge_fit.slope)
        y_intersect = self.bg_fit.evaluate(x_intersect)

        # Display the intersection point
        if self.intersection_point is not None:
            self.plot_widget.removeItem(self.intersection_point)

        self.intersection_point = pg.ScatterPlotItem()
        self.intersection_point.addPoints(x=[x_intersect], y=[y_intersect],
                                          brush=pg.mkBrush('g'), size=10, symbol='o')
        self.plot_widget.addItem(self.intersection_point)

        # Update the value display
        self.intersection_value.setText(f"{x_intersect:.6f}")

    def save_edge(self):
        if not self.intersection_value.text():
            return

        edge_name = self.edge_name_input.text().strip()
        if not edge_name:
            edge_name = f"Edge_{len(self.saved_edges) + 1}"

        energy_value = float(self.intersection_value.text())

        # Save the edge
        self.saved_edges[edge_name] = energy_value

        # Add to the list
        self.saved_edges_list.addItem(f"{edge_name}: {energy_value:.6f} keV")

        # Clear the edge name field
        self.edge_name_input.clear()


if __name__ == '__main__':
    mkQApp("Main")
    main_window = MainWindow()
    main_window.load_data("D:\\OneDrive - Uniwersytet Jagielloński\\Studia\\python\\EELSpy\\pythonProject1\\tests\\reference\\STEM SI_eels_O_Mn_La_do_Ca.dm4")
    main_window.show()
    pg.exec()
