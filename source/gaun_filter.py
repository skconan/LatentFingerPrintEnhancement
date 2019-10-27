import cv2 as cv
import numpy as np
from utilities import *
from fft import * 
from total_variation import *
from enhancement import *

def padding(gray,pad_size):
    rows, cols = gray.shape
    pad = np.zeros((rows+pad_size*2,cols+pad_size*2),np.uint8)
    pad[pad_size:-pad_size,pad_size:-pad_size] = gray
    return pad


if __name__ == "__main__":
    bin = cv.imread(r"E:\KSIP\LatentFingerPrintEnhancement\images\F0000032_bin_block.png",0)
    gray = cv.imread(r"E:\KSIP\LatentFingerPrintEnhancement\images\F0000032.png",0)
    gray = gray[122:122+64,230:230+64]

    bin = cv.resize(bin,(64,64))
    gray = cv.resize(gray,(64,64))
    gray = normalize(gray)

    g = normalize(getGaussianKernel2D((64,64),16,16))
    imshow("gaun",g)
    # g = genGabor((256,256), 0.3, 0, func=np.cos) 
    roi = normalize(gray/255.*g)
    spectrum = spatial2freq(roi)
    
    magnitude = get_magnitude(spectrum)
    magnitude = np.float32(magnitude)
    magnitude = normalize(to_logscale(magnitude))   
    
    imshow("magnitude",magnitude.copy())
    imshow("magnitude_mapping",magnitude.copy(), mapping=True)
    imshow("gray",gray.copy())
    imshow("roi",normalize(roi.copy()))
    

    cv.waitKey(-1)