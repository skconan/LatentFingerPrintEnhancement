import cv2 as cv
from utilities import *
import numpy as np

def zero_mean(data):
    std = data.std()
    mean = data.mean()
    data = np.array(data,np.float16)
    result = (data - mean)/std
    result[result<0] = 0 
    result[result>255] = 255 
    result = normalize(result)
    return result

# def enhancement():
    

# if __name__ == "__main__":
#     enhancement()