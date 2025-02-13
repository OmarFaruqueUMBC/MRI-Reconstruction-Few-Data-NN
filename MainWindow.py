from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import sys
import mainWindowUI
from TrainWindow import TrainWindow 
from TestWindow import TestWindow
from EvaluateWindow import EvaluateWindow


class MainWindow(QtWidgets.QMainWindow, mainWindowUI.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self._train_window = None
        self._test_window = None
        self._evaluate_window = None
        self.buttonTrain.clicked.connect(self.openTrainWindow)
        self.buttonTest.clicked.connect(self.openTestWindow)
        self.buttonEval.clicked.connect(self.openEvaluateWindow)
        


    def openTrainWindow(self):
        self._train_window = TrainWindow()
        self._train_window.show()
        self.close()

    def openTestWindow(self):
        self._test_window = TestWindow()
        self._test_window.show()
        self.close()

    def openEvaluateWindow(self):
        self._evaluate_window = EvaluateWindow()
        self._evaluate_window.show()
        self.close()



def main():
    app = QApplication(sys.argv)
    form = MainWindow()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()