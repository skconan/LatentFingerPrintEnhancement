import cv2 as cv
import numpy as np
from utilities import *

def spatial2freq(img):
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    return img_fft

def get_magnitude(fft2):
    magnitude = cv.magnitude(fft2.real, fft2.imag)
    return magnitude

# def get_phase(fft2):


def to_logscale(img):
    return normalize(20*np.log(img))
    
if __name__ == "__main__":
    img = cv.imread(r"D:\KSIP\Database\00.bmp",0)
    # img = img[100:200,100:200]
    rows, cols = img.shape
    result = np.zeros((rows,cols,3),np.uint8)
    for row in range(0,rows,100):
        for col in range(0,cols,100):
            roi = img[row:row+100, col:col+100]
            img_fft2 = spatial2freq(roi)
            mag = get_magnitude(img_fft2)
            print(img_fft2.shape)

            mag, angle = cv.cartToPolar(img_fft2.real, img_fft2.imag)

            phase = cv.phase(img_fft2.real, img_fft2.imag)

            print("angle",angle)

            mag_mapping = cv.applyColorMap(to_logscale(mag), cv.COLORMAP_JET)
            
            cv.imshow("magnitude",to_logscale(mag))
            cv.imshow("img",img)
            cv.imshow("mag_mapping",mag_mapping)
            result[row:row+100, col:col+100] = mag_mapping
            # cv.imshow("phase",normalize(phase))
            # cv.imshow("polar",polar)
            cv.imshow("result",result)
            cv.waitKey(-1)