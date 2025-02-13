from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import sys
import trainWindowUI
from train import train_net
from time import sleep

class TrainWindow(QtWidgets.QMainWindow, trainWindowUI.Ui_TrainWindow):
    def __init__(self, parent=None):
        super(TrainWindow, self).__init__(parent)
        self.setupUi(self)
        self._main_window = None
        self._test_window = None
        self._evaluation_window = None
        #self.menuMain_Window.triggered.connect(self.openMainWindow)
        self.actionMain.triggered.connect(self.openMainWindow)
        self.actionTest.triggered.connect(self.openTestWindow)
        self.actionEvaluate.triggered.connect(self.openEvaluateWindow)
        self.actionClear.triggered.connect(self.clearInputText)
        self.trainButton.clicked.connect(self.trainNetwork)


    
    def openMainWindow(self):
        from MainWindow import MainWindow
        self._main_window = MainWindow()
        self._main_window.show()
        self.close()

    def openTestWindow(self):
        from TestWindow import TestWindow
        self._test_window = TestWindow()
        self._test_window.show()
        self.close()
        
    def openEvaluateWindow(self):
        from EvaluateWindow import EvaluateWindow
        self._evaluation_window = EvaluateWindow()
        self._evaluation_window.show()
        self.close()

    def clearInputText(self):
        self.imagePath.setText("")
        self.networkPath.setText("")
        self.imageNo.setValue(100)
        self.gpuNo.setValue(0)
        self.iterations.setValue(1000)
        self.msg.setText("")


    def trainNetwork(self):
        self.msg.setText("")
        validate = self.checkInput()
        if (validate != "0"):
            self.msg.setText(validate)
        else:
            self.showProcessing()
            sleep(2)
            imagePath = self.imagePath.text().strip()
            networkPath = self.networkPath.text().strip()
            train_net.functionTrain(
                image_path=imagePath, 
                checkpoints_dir=networkPath, 
                num_imgs=self.imageNo.value(), 
                num_gpus=self.gpuNo.value(), 
                num_epochs=self.iterations.value())
            self.stopShowProcessing()
            self.msg.setText("Trained Network Saved in Network Directory!!!")
            
    def showProcessing(self):
        self.label_7.setText("Processing!!")
        self.label_7.show()
        self.repaint()

    def stopShowProcessing(self):
        self.label_7.setText("")


    def checkInput(self):
        imagePath = self.imagePath.text().strip()
        networkPath = self.networkPath.text().strip()

        if (len(imagePath) == 0):
            return "Path of Raw Image Directory Can't be Empty!!!"
        elif (len(networkPath) == 0):
            return "Path of Network Directory Can't be Empty!!!"
        else:
            return "0"





def main():
    app = QApplication(sys.argv)
    form = TrainWindow()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()