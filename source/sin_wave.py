import cv2 as cv
import numpy as np
from utilities import *
from fft import * 
from total_variation import *
import math

def cosine1D(time, freq):
    # freq = 10
    val_list = []
    # print("freq:",f)
    time = np.array(range(time))/time
    # signal = []
    # for f in freq:
        # signal += list(np.cos(2*np.pi*f*time))
    signal = np.cos(2*np.pi*freq[0]*time) +  np.cos(2*np.pi*freq[1]*time)
    print(signal)
    # signal = np.sin(2*np.pi*f*time)     
    # print(val)
    # val = np.float16(val)
    # val = np.degrees(val)
    # val_list.append(val)
    spectrum  = np.fft.fft(signal)
    # freq = np.fft.fftfreq(10)
    phase = np.angle(spectrum )

    # fft = np.fft.fftshift(fft)
    mag = np.abs(spectrum.copy())

    phase[mag<20] = 0

    plt.title("signal")
    plt.plot(signal)
    # plt.show()
    plt.figure()
    plt.title("spectrum")
    plt.plot(spectrum)
    plt.figure()
    plt.title("magnitude")
    plt.plot(mag)
    plt.figure()
    plt.title("phase")
    plt.plot(phase*180/np.pi)
    plt.show()

def cosine_img(size, freq):
    result = np.ones((size,size),np.uint8)*255
    # freq = 10
    val_list = []
    for x in range(size):
        # val = np.cos(2*np.pi*T*x)
        val = np.cos(2*np.pi*freq[0]*x/size) +  np.cos(2*np.pi*freq[1]*x/size)

        # print(val)
        # val = np.float16(val)
        val_list.append(val)
        val *= 255
        # print(val)
        if val <0 :
            val = 0
        elif val > 255:
            val = 255
        result[:,x] = val
    print(set(val_list))
    img = result
    img_fft2 = spatial2freq(img)
    mag = get_magnitude(img_fft2)
    
    mag_map = cv.applyColorMap(normalize(mag)**2, cv.COLORMAP_HOT)

    # cv.phase(img_fft2.real, img_fft2.imag)
    cv.imshow("mag",mag)
    cv.imshow("mag_map",mag_map)
    cv.imshow("cosine",img)
    cv.waitKey(-1)

def main():
    cosine1D(100,[15,40])
    # cosine_img(500,[50,50])
    
if __name__ == "__main__":
    main()


