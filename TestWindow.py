from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import sys
import testWindowUI
from test import test_net
from time import sleep


class TestWindow(QtWidgets.QMainWindow, testWindowUI.Ui_TestWindow):
    def __init__(self, parent=None):
        super(TestWindow, self).__init__(parent)
        self.setupUi(self)
        self._main_window = None
        self._evaluation_window = None
        self._train_window = None
        self.actionMain.triggered.connect(self.openMainWindow)
        self.actionTrain.triggered.connect(self.openTrainWindow)
        self.actionEvaluate.triggered.connect(self.openEvaluateWindow)
        self.actionClear.triggered.connect(self.clearInputText)
        self.testButton.clicked.connect(self.testNetwork)


    
    def openMainWindow(self):
        from MainWindow import MainWindow
        self._main_window = MainWindow()
        self._main_window.show()
        self.close()

    def openTrainWindow(self):
        from TrainWindow import TrainWindow
        self._train_window = TrainWindow()
        self._train_window.show()
        self.close()
        
    def openEvaluateWindow(self):
        from EvaluateWindow import EvaluateWindow
        self._evaluation_window = EvaluateWindow()
        self._evaluation_window.show()
        self.close()

    def clearInputText(self):
        self.imagePath.setText("")
        self.networkPath.setText("")
        self.resultPath.setText("")
        self.msg.setText("")

    def testNetwork(self):
        self.msg.setText("")
        validate = self.checkInput()
        if (validate != "0"):
            self.msg.setText(validate)
        else:
            self.label_6.setText("Processing!!")   
            self.label_6.show()
            self.repaint()
            sleep(2)
            imagePath = self.imagePath.text().strip()
            networkPath = self.networkPath.text().strip()
            resultPath = self.resultPath.text().strip()
            test_net.eval_diff_plot(
                net_path=networkPath,
                img_path=imagePath,
                substep=int('4'),
                low_freq_percent=float('0.04'),
                results_dir=resultPath,
                exp_name='')
            self.label_6.setText("") 
            self.msg.setText("Test Images Saved in Result Directory!!!")
            



    def checkInput(self):
        imagePath = self.imagePath.text().strip()
        networkPath = self.networkPath.text().strip()
        resultPath = self.resultPath.text().strip()

        if (len(imagePath) == 0):
            return "Path of Raw Image Can't be Empty!!!"
        elif (len(resultPath) == 0):
            return "Path of Result Directory Can't be Empty!!!"
        elif (len(networkPath) == 0):
            return "Path of Network Can't be Empty!!!"
        else:
            return "0"





def main():
    app = QApplication(sys.argv)
    form = TestWindow()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()