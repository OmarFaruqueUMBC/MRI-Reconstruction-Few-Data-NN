# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'evaluate.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_EvaluateWindow(object):
    def setupUi(self, EvaluateWindow):
        EvaluateWindow.setObjectName("EvaluateWindow")
        EvaluateWindow.resize(600, 600)
        self.centralwidget = QtWidgets.QWidget(EvaluateWindow)
        self.centralwidget.setObjectName("centralwidget")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("/DUET.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        EvaluateWindow.setWindowIcon(icon)
        self.imagePath = QtWidgets.QLineEdit(self.centralwidget)
        self.imagePath.setGeometry(QtCore.QRect(220, 140, 350, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.imagePath.setFont(font)
        self.imagePath.setObjectName("imagePath")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 140, 181, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("background-color: rgb(220, 250, 255);")
        self.label_2.setObjectName("label_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(70, 50, 451, 51))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAutoFillBackground(False)
        self.label.setStyleSheet("background-color: rgb(170, 255, 255);")
        self.label.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label.setScaledContents(False)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(30, 260, 181, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("background-color: rgb(220, 250, 255);")
        self.label_4.setObjectName("label_4")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(30, 200, 181, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("background-color: rgb(220, 250, 255);")
        self.label_3.setObjectName("label_3")
        self.networkPath = QtWidgets.QLineEdit(self.centralwidget)
        self.networkPath.setGeometry(QtCore.QRect(220, 260, 350, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.networkPath.setFont(font)
        self.networkPath.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.networkPath.setObjectName("networkPath")
        self.resultPath = QtWidgets.QLineEdit(self.centralwidget)
        self.resultPath.setGeometry(QtCore.QRect(220, 200, 350, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.resultPath.setFont(font)
        self.resultPath.setObjectName("resultPath")
        self.evaluateButton = QtWidgets.QPushButton(self.centralwidget)
        self.evaluateButton.setGeometry(QtCore.QRect(420, 380, 150, 40))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.evaluateButton.setFont(font)
        self.evaluateButton.setStyleSheet("background-color: rgb(0, 170, 0);\n"
"color: rgb(255, 255, 255);")
        self.evaluateButton.setObjectName("evaluateButton")
        self.msg = QtWidgets.QLabel(self.centralwidget)
        self.msg.setGeometry(QtCore.QRect(30, 380, 340, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.msg.setFont(font)
        self.msg.setStyleSheet("color: rgb(255, 0, 0);")
        self.msg.setText("")
        self.msg.setObjectName("msg")
        self.expName = QtWidgets.QLineEdit(self.centralwidget)
        self.expName.setGeometry(QtCore.QRect(220, 320, 350, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.expName.setFont(font)
        self.expName.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.expName.setObjectName("expName")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(30, 320, 181, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("background-color: rgb(220, 250, 255);")
        self.label_5.setObjectName("label_5")

        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(420, 430, 150, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.label_6.setFont(font)
        self.label_6.setStyleSheet("background-color: rgb(220, 250, 255);color: rgb(255, 0, 0);")
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")

        EvaluateWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(EvaluateWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 600, 21))
        self.menubar.setObjectName("menubar")
        self.menuMain_Window = QtWidgets.QMenu(self.menubar)
        self.menuMain_Window.setObjectName("menuMain_Window")
        self.actionMain = QtWidgets.QAction(EvaluateWindow)
        self.actionMain.setObjectName("actionMain")
        self.menuMain_Window.addAction(self.actionMain)

        self.menuTrain_CNN = QtWidgets.QMenu(self.menubar)
        self.menuTrain_CNN.setObjectName("menuTrain_CNN")
        self.actionTrain = QtWidgets.QAction(EvaluateWindow)
        self.actionTrain.setObjectName("actionTrain")
        self.menuTrain_CNN.addAction(self.actionTrain)

        self.menuTeat_CNN = QtWidgets.QMenu(self.menubar)
        self.menuTeat_CNN.setObjectName("menuTeat_CNN")
        self.actionTest = QtWidgets.QAction(EvaluateWindow)
        self.actionTest.setObjectName("actionTest")
        self.menuTeat_CNN.addAction(self.actionTest)

        self.menuClear_Input = QtWidgets.QMenu(self.menubar)
        self.menuClear_Input.setObjectName("menuClear_Input")
        self.actionClear = QtWidgets.QAction(EvaluateWindow)
        self.actionClear.setObjectName("actionClear")
        self.menuClear_Input.addAction(self.actionClear)

        EvaluateWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(EvaluateWindow)
        self.statusbar.setObjectName("statusbar")
        EvaluateWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuMain_Window.menuAction())
        self.menubar.addAction(self.menuTrain_CNN.menuAction())
        self.menubar.addAction(self.menuTeat_CNN.menuAction())
        self.menubar.addAction(self.menuClear_Input.menuAction())

        self.retranslateUi(EvaluateWindow)
        QtCore.QMetaObject.connectSlotsByName(EvaluateWindow)

    def retranslateUi(self, EvaluateWindow):
        _translate = QtCore.QCoreApplication.translate
        EvaluateWindow.setWindowTitle(_translate("EvaluateWindow", "Reconstruct MRI"))
        self.label_2.setStatusTip(_translate("EvaluateWindow", "Please write directory psth of a raw image!!"))
        self.label_2.setText(_translate("EvaluateWindow", "<html><head/><body><p align=\"center\">Path of Raw Image</p></body></html>"))
        self.label.setText(_translate("EvaluateWindow", "Evaluate CNN for MRI Reconstruction"))
        self.label_4.setText(_translate("EvaluateWindow", "<html><head/><body><p align=\"center\">Path of Trained Network</p></body></html>"))
        self.label_3.setText(_translate("EvaluateWindow", "<html><head/><body><p align=\"center\">Path To Save Result</p></body></html>"))
        self.evaluateButton.setText(_translate("EvaluateWindow", "Reconstruct"))
        self.label_5.setText(_translate("EvaluateWindow", "<html><head/><body><p align=\"center\">Experiment Name</p></body></html>"))
        self.menuMain_Window.setTitle(_translate("EvaluateWindow", "Main Window"))
        self.menuTrain_CNN.setTitle(_translate("EvaluateWindow", "Train CNN"))
        self.menuTeat_CNN.setTitle(_translate("EvaluateWindow", "Test CNN"))
        self.menuClear_Input.setTitle(_translate("EvaluateWindow", "Clear Input"))
        self.actionMain.setText(_translate("EvaluateWindow", "Open"))
        self.actionTest.setText(_translate("EvaluateWindow", "Open"))
        self.actionTrain.setText(_translate("EvaluateWindow", "Open"))
        self.actionClear.setText(_translate("EvaluateWindow", "Clear"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    EvaluateWindow = QtWidgets.QMainWindow()
    ui = Ui_EvaluateWindow()
    ui.setupUi(EvaluateWindow)
    EvaluateWindow.show()
    sys.exit(app.exec_())
