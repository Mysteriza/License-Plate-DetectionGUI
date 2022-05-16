import sys
import cv2
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
import pandas as pd
import numpy as np
from tkinter import filedialog as fd
import time
import pytesseract


def konvolusi(X, F):
    X_height = X.shape[0]
    X_width = X.shape[1]
    F_height = F.shape[0]
    F_width = F.shape[1]
    H = (F_height) // 2
    W = (F_width) // 2
    out = np.zeros((X_height, X_width))
    for i in np.arange(H + 1, X_height - H):
        for j in np.arange(W + 1, X_width - W):
            sum = 0
            for k in np.arange(-H, H + 1):
                for l in np.arange(-W, W + 1):
                    a = X[i + k, j + l]
                    w = F[H + k, W + l]
                    sum = sum + (w * a)
            out[i, j] = sum
    return out

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUI.ui', self)
        self.Image = None
        self.Image2 = None

        # file
        self.actionOpen.triggered.connect(self.load)
        self.actionSave.triggered.connect(self.save)

        # GUI button
        self.deteksiButton.clicked.connect(self.detectPlat)
        self.actionReset.triggered.connect(self.resetButt)
        self.loadButton.clicked.connect(self.loadCitra)

        # Operasi Titik
        self.action_Grayscale.triggered.connect(self.gray)
        self.actionOperasi_Pencerahan.triggered.connect(self.pencerahan)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Stretching.triggered.connect(self.kontrasStretching)
        self.actionNegativeImage.triggered.connect(self.negativeImage)
        self.actionBiner_Image.triggered.connect(self.binerImage)

        #Slider Kontras
        self.horizontalSlider.valueChanged.connect(self.sliderKecerahan)

    # Tampilkan Gambar
    def load(self):
        path = fd.askopenfilename()
        if path:
            self.temp = path.split('.jpg')
            self.temp = self.temp[0]
            self.gambar = cv2.imread(path)
            self.Image = self.gambar
            self.displayImage(1)
        else:
            print("Gagal Memuat!")

    def loadCitra(self):
        path = fd.askopenfilename()
        if path:
            self.temp = path.split('.jpg')
            self.temp = self.temp[0]
            self.gambar = cv2.imread(path)
            self.Image = self.gambar
            self.displayImage(1)
        else:
            print("Gagal Memuat!")

    def save(self):
        image, filter = QFileDialog.getSaveFileName(self, 'Save File', 'D:\\', "Image Files (*.jpg)")
        if image:
            cv2.imwrite(image, self.Image)
        else:
            print('Error')

    def resetButt(self):
        self.imgLabel.clear()
        self.imgLabel_gray.clear()
        self.imgLabel_bilateral.clear()
        self.imgLabel_canny.clear()
        self.hasilLabel.clear()
        self.OutputLabel.clear()

    def gray(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        self.displayImage(2)
        df = pd.DataFrame(self.Image)
        df.to_csv('Gray.csv', mode='a')

    def pencerahan(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass
        H, W = self.Image.shape[:2]
        pencerahan = 50
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a + pencerahan, 0, 255)

                self.Image.itemset((i, j), b)
        self.displayImage(2)
        df = pd.DataFrame(self.Image)
        df.to_csv('Pencerahan.csv', mode='a')

    def contrast(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass
        H, W = self.Image.shape[:2]
        contrast = 1.6
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a * contrast, 0, 255)

                self.Image.itemset((i, j), b)
        self.displayImage(2)
        df = pd.DataFrame(self.Image)
        df.to_csv('Kontras.csv', mode='a')

    def kontrasStretching(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass
        H, W = self.Image.shape[:2]
        minV = np.min(self.Image)
        maxV = np.max(self.Image)
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = float(a - minV) / (maxV - minV) * 255

                self.Image.itemset((i, j), b)
        self.displayImage(2)
        df = pd.DataFrame(self.Image)
        df.to_csv('Kontras Stretching.csv', mode='a')

    def negativeImage(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass
        H, W = self.Image.shape[:2]
        max = 255
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = float(255 - a)

                self.Image.itemset((i, j), b)
        self.displayImage(2)
        df = pd.DataFrame(self.Image)
        df.to_csv('Negative Image.csv', mode='a')

    def binerImage(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        threshold = 180
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                if a > threshold:
                    b = 255
                if a < threshold:
                    b = 1
                if a == threshold:
                    b = 0

                self.Image.itemset((i, j), b)
        self.displayImage(2)
        df = pd.DataFrame(self.Image)
        df.to_csv('Biner Image.csv', mode='a')


    def detectPlat(self):
        # Konversi ke grayscale
        gray = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        self.Image2 = self.Image
        self.Image = gray
        self.displayImage(3)

        #Blur untuk mereduksi noise
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        self.Image = bilateral
        self.displayImage(4)

        # Lakukan deteksi tepi
        edged = cv2.Canny(gray, 170, 200)
        self.Image = edged
        self.displayImage(5)

        # Cari kontur pada tepi gambar
        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

        NumberPlateCnt = None
        count = 0
        # Loop kontur
        for c in cnts:
            # Perkirakan konturnya
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.06 * peri, True)
            # jika kontur yang diperkirakan memiliki empat titik, maka asumsikan bahwa layar ditemukan
            if len(approx) == 4:
                NumberPlateCnt = approx
                break

        # mask bagian selain plat nomor
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
        new_image = cv2.bitwise_and(self.Image2, self.Image2, mask=mask)
        self.Image = new_image
        self.displayImage(2)

        # konfigurasi untuk tesseract
        config = '-l eng --oem 1 --psm 3'

        # jalankan tesseract OCR pada gambar
        text = pytesseract.image_to_string(new_image, config=config)
        print("Plat Nomornya Adalah: ", text)
        if text == "":
            self.OutputLabel.setText("Plat Nomor Tidak Terdeteksi !")
        else:
            self.OutputLabel.setText("Plat Nomor Terdeteksi !")


        # data disimpan dalam file CSV
        raw_data = {'Tanggal Pendeteksian: ': [time.asctime(time.localtime(time.time()))], '': [text]}
        df = pd.DataFrame(raw_data)
        df.to_csv('Data Plat Nomor Yang Terdeteksi.csv', mode='a')


    def sliderKecerahan(self, value):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass
        H, W = self.Image.shape[:2]
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a + value, 0, 255)
                self.Image.itemset((i, j), b)
        self.displayImage(2)

    def displayImage(self, layar):
        qformat = QImage.Format_Indexed8

        if len(self.Image.shape) == 3:
            if (self.Image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0], self.Image.strides[0], qformat)

        img = img.rgbSwapped()
        if layar == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.imgLabel.setScaledContents(True)

        elif layar == 2:
            self.hasilLabel.setPixmap(QPixmap.fromImage(img))
            self.hasilLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.hasilLabel.setScaledContents(True)

        elif layar == 3:
            self.imgLabel_gray.setPixmap(QPixmap.fromImage(img))
            self.imgLabel_gray.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.imgLabel_gray.setScaledContents(True)

        elif layar == 4:
            self.imgLabel_bilateral.setPixmap(QPixmap.fromImage(img))
            self.imgLabel_bilateral.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.imgLabel_bilateral.setScaledContents(True)

        elif layar == 5:
            self.imgLabel_canny.setPixmap(QPixmap.fromImage(img))
            self.imgLabel_canny.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.imgLabel_canny.setScaledContents(True)


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Perbaikan Citra')
window.show()
sys.exit(app.exec_())
