import cv2 as cv
import numpy as np
from utilities import *
from fft import * 
from total_variation import *

if __name__ == "__main__":
    img = cv.imread(r"D:\KSIP\Database\00.bmp")
    # img = cv.imread(r"D:\KSIP\Database\NIST27\Latent\B101L9U.bmp")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rows, cols,ch = img.shape
    result = img.copy()
    result1 = img.copy()
    step = 80
    plt.title("original");plt.hist(gray.ravel(),256,[0,256]); plt.show()
    gray = zero_mean(gray.copy())
    plt.title("zero mean");plt.hist(gray.ravel(),256,[0,256]); plt.show()

    print(gray.max(),gray.min())
    cv.imshow("gray",gray)

    for row in range(0,rows,step):
        for col in range(0,cols,step):
            roi = gray[row:row+step, col:col+step]
            roi_img = img[row:row+step, col:col+step].copy()
            cv.imshow("roi",roi)
            spectrum = spatial2freq(roi)
            # spectrum[24:26,24:26] = 0
            mag = get_magnitude(spectrum)
            phase = get_phase(spectrum) * 180 / np.pi

            print("Phase",phase.min(),phase.max())
            print("Magnitude",mag.min(),mag.max())

            phase = normalize(phase,input_max=180,input_min=-180)
            phase[phase<200] = 0
            phase = cv.applyColorMap(phase, cv.COLORMAP_HSV)

            img_filtered = freq2spatial(spectrum)

            mag_mapping = cv.applyColorMap(normalize(to_logscale(mag)), cv.COLORMAP_HOT)
            img_filtered = cv.cvtColor(img_filtered,cv.COLOR_GRAY2BGR)
            cv.imshow("magnitude",to_logscale(mag**2))
            cv.imshow("img",img)
            cv.imshow("mag_mapping",cv.resize(mag_mapping,None,fx=2,fy=2))
            result[row:row+step, col:col+step] = cv.addWeighted(mag_mapping,0.7,img[row:row+step, col:col+step],0.3,0)
            cv.imshow("phase",phase)
            cv.imshow("result",result)
            cv.waitKey(-1)

    cv.imwrite("./result.png",result)

