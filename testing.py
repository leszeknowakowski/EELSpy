import  hyperspy.api as hs
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets, mkQApp
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication,QWidget,QVBoxLayout
from PyQt5.QtGui import QCursor, QFont
font = {'color': 'b', 'font-size': '14pt'}
my_font = QFont("Times", 10, QFont.Bold)
my_font.setBold(True)

def set_plot_fonts(plot, color='w', font="Times", size=10, bold=False):
    for label in ['left', 'right', 'top', 'bottom']:
        my_font = QFont(font, size, QFont.Bold)
        plot.getAxis(label).setTickFont(my_font)
        plot.getAxis(label).setTextPen(color)
class ProcessFile:
    def __init__(self, file):
        s = hs.load(file)
        self.haadf = s[0]
        self.edx = s[1]

    def decompose_into_lines(self):
        self.images = self.edx.get_lines_intensity()
        return self.images

    def get_lines(self):
        lines = self. edx.metadata.Sample.xray_lines
        return

    def get_spectral_resolution(self):
        res = self.edx.axes_manager["Energy"].scale
        return res


class MainWindow(QtWidgets.QMainWindow):
    """ Example application main window """

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.graphics_window = pg.GraphicsLayoutWidget(show=True)
        self.setCentralWidget(self.graphics_window)
        self.setWindowTitle('EELSpy')
        self.resize(600, 500)
        self.show()

        pg.setConfigOption('imageAxisOrder', 'row-major')  # Switch default order to Row-major
        self.file="F:\\OneDrive - Uniwersytet Jagielloński\\Studia\\pomiary\\TEM\\Co3O4-CeO2\\Co3O4_CeO2_1_1\\map1.bcf"
        self.data = ProcessFile(file)

    def create_matrix(self,data):
        # Example correlation matrix (replace with Co_ka)
        # self.corrMatrix = Co_ka
        self.corrMatrix = data
        self.correlogram = pg.ImageItem()
        tr = QtGui.QTransform().translate(-0.5, -0.5)
        self.correlogram.setTransform(tr)
        self.correlogram.setImage(self.corrMatrix)
        self.correlogram.mouseClickEvent = self.on_map_left_clicked
        return self.correlogram

    def add_plot(self, item, name=""):
        self.plotItem = self.graphics_window.addPlot()
        self.plotItem.setLabel("left", "y dimension", **font)  # Y-axis label with unit
        self.plotItem.setLabel("bottom", "x dimension",**font)  # X-axis label with unit
        #self.plotItem.getAxis("bottom").label.setFont(my_font)
        self.plotItem.invertY(True)
        self.plotItem.setDefaultPadding(0.0)
        self.plotItem.addItem(item)

        self.plotItem.showAxes(True, showValues=(True, True, False, False), size=20)
        self.plotItem.getAxis('bottom').setHeight(10)

        colorMap = pg.colormap.get("CET-D1")
        bar = pg.ColorBarItem(values=(-1, 1), colorMap=colorMap)
        bar.setImageItem(item, insert_in=self.plotItem)

        return

    def on_map_left_clicked(self, event):
        if event.button() == 1 and event.modifiers() == Qt.ShiftModifier:
            pos = event.pos()
            x = int(pos.x())
            y = int(pos.y())
            ydata = self.data.edx.data[x,y]

            screen_rect = QApplication.primaryScreen().geometry()
            pos = QCursor.pos()
            spectrum_res = self.data.get_spectral_resolution()
            x = np.array(range(len(ydata))) * spectrum_res
            self.scf_window = SpectrumPlot(x, ydata, pos, screen_rect)
            self.scf_window.show()
    def add_line_to_plot(self, line):
        pass

class SpectrumPlot(QWidget):
    """Window that shows a SCF convergence plot"""
    def __init__(self,xaxis, energies, pos, screen_rect):
        super().__init__()
        self.setWindowTitle("Random Line Plot")
        self.resize(800, 600)  # Fixed window size

        win_width, win_height = self.width(), self.height()

        # Adjust position to keep window within screen boundaries
        x = min(pos.x() + 20, screen_rect.right() - win_width)  # Prevent right overflow
        y = min(pos.y() + 20, screen_rect.bottom() - win_height)  # Prevent bottom overflow

        x = max(x, screen_rect.left())  # Prevent left overflow
        y = max(y, screen_rect.top())  # Prevent top overflow

        self.setGeometry(x, y, 800, 600)

        layout = QVBoxLayout()
        self.plot_widget = pg.PlotWidget(background='w')
        self.plot_widget.setLabel("left", "counts", **font)  # Y-axis label with unit
        self.plot_widget.setLabel("bottom", "energy", units="keV", **font)  # X-axis label with unit
        set_plot_fonts(self.plot_widget, size=11, color='k')
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

        self.plot_widget.plot(xaxis, energies, pen='k')



if __name__ == '__main__':
    mkQApp("Main")
    file = "F:\\OneDrive - Uniwersytet Jagielloński\\Studia\\pomiary\\TEM\\Co3O4-CeO2\\Co3O4_CeO2_1_1\\map1.bcf"
    data = ProcessFile(file)
    element_data = data.decompose_into_lines()
    spectrum_data = element_data[2].data

    main_window = MainWindow()
    item = main_window.create_matrix(spectrum_data)
    main_window.add_plot(item, name="mainw")
    pg.exec()
