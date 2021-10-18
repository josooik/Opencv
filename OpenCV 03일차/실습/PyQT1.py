# PyQT 컬러이미지 -> 그레이스케일 이미지 변환 후 출력
from PyQt5 import QtCore, QtGui
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QImage

import sys
import cv2
import numpy as np

class Ui(QtWidgets.QDialog):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('PyQT/img_proc_qt1.ui', self)
        self.btn_load = self.findChild(QtWidgets.QPushButton, 'btn_load')
        self.btn_load.clicked.connect(self.btn_load_clicked)
        self.btn_run = self.findChild(QtWidgets.QPushButton, 'btn_run')
        self.btn_run.clicked.connect(self.btn_run_clicked)
        self.lbl_src = self.findChild(QtWidgets.QLabel, 'lbl_src')
        self.lbl_dst = self.findChild(QtWidgets.QLabel, 'lbl_dst')
        self.line_edit = self.findChild(QtWidgets.QLineEdit, 'line_edit')
        self.line_edit.clear()
        self.show()

    # cv2.imread가 한글 지원하지 않으므로 새로운 방식으로 파일 조합
    def imread(self, filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
        try:
            n = np.fromfile(filename, dtype)
            img = cv2.imdecode(n, flags)
            return img

        except Exception as e:
            print(e)
            return None


    def btn_load_clicked(self):
        path = 'img'
        filter = "All Images(*.jpg; *.png; *.bmp);;JPG (*.jpg);;PNG(*.png);;BMP(*.bmp)"
        fname = QtWidgets.QFileDialog.getOpenFileName(self, "파일로드", path, filter)
        filename = str(fname[0])
        self.line_edit.setText(filename)
        self.img_src = self.imread(filename)
        self.display_output_image(self.img_src, 0)

    def btn_run_clicked(self):
        self.img_gray = cv2.cvtColor(self.img_src, cv2.COLOR_BGR2GRAY)
        self.display_output_image(self.img_gray, 1)


    def display_output_image(self, img, mode):
        h, w = img.shape[:2] # 그레이영상의 경우 ndim이 2이므로 h, w, ch 형태로 값을 얻어올수 없다.
        
        if img.ndim == 2:
            qimg = QImage(img, w, h, w * 1, QImage.Format_Grayscale8)

        else:
            bytes_per_line = img.shape[2] * w
            qimg = QImage(img, w, h, bytes_per_line, QImage.Format_BGR888)
            
        pixmap = QtGui.QPixmap(qimg)
        pixmap = pixmap.scaled(600, 450, QtCore.Qt.KeepAspectRatio)    # 이미지 비율유지
        #pixmap = self.pixmap.scaled(600, 450, QtCore.Qt.lgnoreAspectRatio)  # 이미지를 프레임에 맞춤
        
        if mode == 0:
            self.lbl_src.setPixmap(pixmap)
            self.lbl_src.update() # 프레임 띄우기
        
        else:
            self.lbl_dst.setPixmap(pixmap)
            self.lbl_dst.update() # 프레임 띄우기


app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()