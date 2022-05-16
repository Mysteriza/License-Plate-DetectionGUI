import time

import cv as cv
import cv2
import imutils
import numpy as np
import pandas as pd
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Baca dan atur ukuran gambar
image = cv2.imread('Plat 1.jpg')
image = imutils.resize(image, width=500)
#cv2.imshow("Gambar Original", image)

# Konversi ke grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gambar Grayscale", gray)

# Blur untuk mereduksi noise
bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("Bilateral Filter", bilateral)

kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(bilateral, cv2.MORPH_OPEN, kernel)
cv2.imshow("Opening", opening)


gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)
cv2.imshow("Gradient", gradient)

# Lakukan deteksi tepi
edged = cv2.Canny(gradient, 170, 200)
cv2.imshow("Canny Edges", edged)


# Cari kontur pada tepi gambar
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

NumberPlateCnt = None
#count = 0
# Loop kontur
for c in cnts:
    # Perkirakan konturnya
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # jika kontur yang diperkirakan memiliki empat titik, maka asumsikan bahwa layar ditemukan
    if len(approx) == 4:
        NumberPlateCnt = approx
        break

cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)

# mask bagian selain plat nomor
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
new_image = cv2.bitwise_and(image, image, mask=mask)
cv2.namedWindow("Gambar Final", cv2.WINDOW_NORMAL)
cv2.imshow("Gambar Final", new_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# konfigurasi untuk tesseract
config = ('-l eng --oem 1 --psm 3')

# jalankan tesseract OCR pada gambar
text = pytesseract.image_to_string(new_image, config=config)
print(" Plat Nomornya Adalah: ", text)

# data disimpan dalam file CSV
raw_data = {'Tanggal Pendeteksian : ': [time.asctime(time.localtime(time.time()))], '': [text]}
df = pd.DataFrame(raw_data)
df.to_csv('Data Plat Nomor Yang Terdeteksi.csv', mode='a')

cv2.waitKey(0)
cv2.destroyAllWindows()
