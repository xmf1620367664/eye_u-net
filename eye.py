# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'eye.ui'
#
# Created by: PyQt5 UI code generator 5.14.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow,QApplication,QTextEdit,QAction,QFileDialog,QWidget,QGraphicsScene,QGraphicsPixmapItem
from PyQt5.QtGui import QIcon,QImage,QPixmap
from PIL import Image
import numpy as np
import cv2
import os
from pred import Pred


class Ui_MainWindow(QWidget):
    def __init__(self,name='Ui_MainWindow'):
        super(Ui_MainWindow,self).__init__()
        self.cwd=os.getcwd()
        self.pred=Pred()
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1148, 835)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(290, 40, 531, 121))
        font = QtGui.QFont()
        font.setFamily("DFPYuanW3-GB5")
        font.setPointSize(26)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(110, 200, 931, 391))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.line_2 = QtWidgets.QFrame(self.widget)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.horizontalLayout.addWidget(self.line_2)
        self.graphicsView = QtWidgets.QGraphicsView(self.widget)
        self.graphicsView.setObjectName("graphicsView")
        self.horizontalLayout.addWidget(self.graphicsView)
        self.line = QtWidgets.QFrame(self.widget)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout.addWidget(self.line)
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.widget)
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.horizontalLayout.addWidget(self.graphicsView_2)
        self.line_3 = QtWidgets.QFrame(self.widget)
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.horizontalLayout.addWidget(self.line_3)
        self.widget1 = QtWidgets.QWidget(self.centralwidget)
        self.widget1.setGeometry(QtCore.QRect(430, 650, 611, 81))
        font = QtGui.QFont()
        font.setFamily("AXIS Std R")
        font.setPointSize(14)
        font.setUnderline(False)
        self.widget1.setFont(font)
        self.widget1.setObjectName("widget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget1)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.lineEdit = QtWidgets.QLineEdit(self.widget1)
        font = QtGui.QFont()
        font.setFamily("AXIS Std R")
        font.setPointSize(14)
        font.setUnderline(False)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_2.addWidget(self.lineEdit)
        self.pushButton = QtWidgets.QPushButton(self.widget1)
        font = QtGui.QFont()
        font.setFamily("AXIS Std R")
        font.setPointSize(14)
        font.setUnderline(False)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.widget1)
        font = QtGui.QFont()
        font.setFamily("AXIS Std R")
        font.setPointSize(14)
        font.setUnderline(False)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.widget1)
        font = QtGui.QFont()
        font.setFamily("AXIS Std R")
        font.setPointSize(14)
        font.setUnderline(False)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_2.addWidget(self.pushButton_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1148, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.pushButton.clicked.connect(self.choose_file)
        self.pushButton.clicked.connect(self.graphicsView.show)
        self.pushButton.clicked.connect(self.graphicsView.show)
        self.pushButton_2.clicked.connect(self.lineEdit.clear)
        self.pushButton_2.clicked.connect(self.graphicsView_2.invalidateScene)
        self.pushButton_2.clicked.connect(self.graphicsView.clearMask)

        self.pushButton_3.clicked.connect(self.show_result)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def choose_file(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,
                                                                "选取文件",
                                                                self.cwd,  # 起始路径
                                                                "JPG Files (*.jpg);;PNG Files (*.png);;TIF Files (*.tif)")  # 设置文件扩展名过滤,用双分号间隔

        if fileName_choose == "":
            print("\n取消选择")
            self.lineEdit.setText("取消选择")
            return

        print("\n你选择的文件为:")
        self.lineEdit.setText(fileName_choose)
        print(fileName_choose)
        print("文件筛选器类型: ", filetype)

        img = cv2.imread(fileName_choose)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img=Image.open(fileName_choose)
        #img=img.resize([224,224])
        #img.show()
        #获得控件尺寸==》（0，0，449，391）
        print(self.graphicsView.rect())
        img=np.uint8(img)
        self.img=img
        x = img.shape[1]
        y = img.shape[0]
        min=391
        scale=min/max(x,y)
        self.zoomscale = scale
        frame = QImage(img, x, y,x*img.shape[2], QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)
        self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)


        # self.graphicsView.show()

    def show_result(self):
        result = self.pred.gui_predict(self.img)
        # print(result.shape)
        result = np.reshape(result, [584, 565])
        new_result = np.ones([584, 565, 3])
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if result[i][j] == 0:
                    new_result[i][j] = [0, 0, 0]
                else:
                    new_result[i][j] = [255, 255, 255]
        img = np.uint8(new_result)
        x = img.shape[1]
        y = img.shape[0]
        min = 391
        scale = min / max(x, y)
        self.zoomscale = scale
        frame = QImage(img, x, y, x * img.shape[2], QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)
        self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()
        self.scene.addItem(self.item)
        self.graphicsView_2.setScene(self.scene)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600; color:#280079;\">眼底视网膜血管分割应用程序</span></p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "选择"))
        self.pushButton_2.setText(_translate("MainWindow", "重置"))
        self.pushButton_3.setText(_translate("MainWindow", "提交"))
