import  hyperspy.api as hs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')
matplotlib.rcParams["backend"] = "Agg"
s = hs.load("D:\\OneDrive - Uniwersytet Jagielloński\\Studia\\pomiary\\TEM\\Co3O4-CeO2\\Co3O4_CeO2_1_1\\map1.bcf")
haadf = s[0]
edx = s[1]
im = edx.get_lines_intensity()

Co_ka = im[2].data

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets, mkQApp


class MainWindow(QtWidgets.QMainWindow):
    """ example application main window """

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        gr_wid = pg.GraphicsLayoutWidget(show=True)
        self.setCentralWidget(gr_wid)
        self.setWindowTitle('pyqtgraph example: Correlation matrix display')
        self.resize(600, 500)
        self.show()

        corrMatrix = Co_ka
        columns = ["A", "B", "C"]

        pg.setConfigOption('imageAxisOrder', 'row-major')  # Switch default order to Row-major

        correlogram = pg.ImageItem()
        # create transform to center the corner element on the origin, for any assigned image:
        tr = QtGui.QTransform().translate(-0.5, -0.5)
        correlogram.setTransform(tr)
        correlogram.setImage(corrMatrix)

        plotItem = gr_wid.addPlot()  # add PlotItem to the main GraphicsLayoutWidget
        plotItem.invertY(True)  # orient y axis to run top-to-bottom
        plotItem.setDefaultPadding(0.0)  # plot without padding data range
        plotItem.addItem(correlogram)  # display correlogram

        # show full frame, label tick marks at top and left sides, with some extra space for labels:
        plotItem.showAxes(True, showValues=(True, True, False, False), size=20)

        # define major tick marks and labels:
        ticks = [(idx, label) for idx, label in enumerate(columns)]
        for side in ('left', 'top', 'right', 'bottom'):
            plotItem.getAxis(side).setTicks((ticks, []))  # add list of major ticks; no minor ticks
        plotItem.getAxis('bottom').setHeight(10)  # include some additional space at bottom of figure

        colorMap = pg.colormap.get("CET-D1")  # choose perceptually uniform, diverging color map
        # generate an adjustabled color bar, initially spanning -1 to 1:
        bar = pg.ColorBarItem(values=(-1, 1), colorMap=colorMap)
        # link color bar and color map to correlogram, and show it in plotItem:
        bar.setImageItem(correlogram, insert_in=plotItem)
        correlogram.mouseClickEvent = self.get_mouse_position


mkQApp("Correlation matrix display")
main_window = MainWindow()

## Start Qt event loop
if __name__ == '__main__':
    pg.exec()