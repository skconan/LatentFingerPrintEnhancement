import cv2 as cv
from utilities import *
import imutils
import matplotlib.pyplot as plt
from fft import *


def nothing(val):
    pass


def frequency_estimation(img, kernel=64, step=16):
    rows, cols = img.shape
    result_ff = np.zeros((rows, cols), np.float32)
    cir1 = np.zeros((kernel, kernel), np.uint8)
    cir2 = np.zeros((kernel, kernel), np.uint8)
    cv.circle(cir1, (kernel//2, kernel//2), 2, (255, 255, 255), -1)
    cv.circle(cir2, (kernel//2, kernel//2), 15, (255, 255, 255), -1)
    bandpass = cir2 - cir1

    for r in range(kernel//2, rows-kernel//2, step):
        for c in range(kernel//2, cols-kernel//2, step):
            roi = img[r-kernel//2:r+kernel//2, c-kernel//2:c+kernel//2].copy()
            spectrum = spatial2freq(roi)

            magnitude = get_magnitude(spectrum)
            magnitude = magnitude_amplification(magnitude)
            magnitude = normalize(magnitude)
            magnitude = cv.bitwise_and(magnitude, bandpass)

            y, x = np.where(magnitude == magnitude.max())

            if len(x) >= 2:
                p1 = x[0], y[0]
                p2 = x[1], y[1]

                freq = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)/2.

                d = freq/kernel

                if 2 <= freq <= 5 and np.var(roi) >= 500:
                    result_ff[r-step//2:r+step//2, c-step//2:c+step//2] = 255
                else:
                    result_ff[r-step//2:r+step//2, c-step//2:c+step//2] = 0
            else:
                result_ff[r-step//2:r+step//2, c-step//2:c+step//2] = 0

    return normalize(result_ff)


def thresholding():
    # dir_sd27 = r"E:\OneDrive\KSIP\Database\NIST27\Latent"

    # img_path = dir_sd27 + "\G002L3U.bmp"
    dir_nist14 = r"E:\OneDrive\KSIP\Database\NIST14"
    img_path = dir_nist14 + r"\F0000001.png"
    gray = cv.imread(img_path, 0)
    # gray = imutils.resize(gray, width=500)
    plt.ion()
    cv.namedWindow("gray")

    cv.createTrackbar("i_min", "gray", 0, 255, nothing)
    cv.createTrackbar("i_max", "gray", 0, 255, nothing)

    result_ff = frequency_estimation(gray, kernel=32, step=8)

    gray = np.float32(gray)
    gray = (gray - gray.mean())

    gray = np.clip(gray, 0, 255)
    gray = np.uint8(gray)
    # clahe = cv.createCLAHE(clipLimit=4, tileGridSize=(10, 10))
    # print("ClipLimit", clahe.getClipLimit())
    # gray = clahe.apply(gray)

    # gray = cv.equalizeHist(gray)

    cv.imshow("original", gray)
    plt.ion()

    cv.imshow("result_ff", result_ff)

    while True:
        i_min = cv.getTrackbarPos("i_min", "gray")
        i_max = cv.getTrackbarPos("i_max", "gray")

        lowerb = np.array([i_min], np.uint8)
        upperb = np.array([i_max], np.uint8)
        mask = cv.inRange(gray, lowerb, upperb)

        th_result = cv.bitwise_and(gray, gray, mask=mask)
        # img_result = cv.bitwise_and(img, img, mask=mask)

        hist = cv.calcHist([gray], [0], None, [256], [0, 256])
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.pause(0.0001)
        plt.cla()

        # cv.imshow("mask", mask)
        cv.imshow("gray", mask)

        k = cv.waitKey(1) & 0xff

        if k == ord('q'):
            break
    plt.show()


def main():
    img_path = r"E:\OneDrive\KSIP\LatentFingerPrintEnhancement\images\F0000032.png"
    dir_sd27 = r"E:\OneDrive\KSIP\Database\NIST27\Latent"

    img_path = dir_sd27 + "\G002L3U.bmp"
    img = cv.imread(img_path)
    clahe = apply_clahe(img)
    cv.imshow("clahe", clahe)
    cv.imshow("original", img)
    cv.waitKey(-1)


if __name__ == "__main__":
    thresholding()
