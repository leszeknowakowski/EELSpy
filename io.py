import  hyperspy.api as hs
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets, mkQApp
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication,QWidget,QVBoxLayout
from PyQt5.QtGui import QCursor, QFont
from testing import EDXSpectrum, HAADFImage, EELSSpectrum

def load_file(fname):
    loaded = hs.load(file)