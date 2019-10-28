import cv2 as cv
import numpy as np
from utilities import *
from matplotlib import pyplot as plt

def spatial2freq(img):
    spectrum = np.fft.fft2(img)
    # print(spectrum[0,0])
    spectrum = np.fft.fftshift(spectrum)
    return spectrum

def magnitude_amplication(magnitude):
    rows,cols = magnitude.shape
    magnitude[rows//2,cols//2] = 0
    magnitude = magnitude - magnitude.mean()
    magnitude[magnitude<0] = 0
    return magnitude
    
def freq2spatial(spectrum):
    spectrum  = np.fft.ifftshift(spectrum )
    img  = np.fft.ifft2(spectrum )
    img = get_magnitude(img)
    img = normalize(img)
    return img

def get_magnitude(spectrum):
    # magnitude = np.abs(spectrum)
    magnitude = np.sqrt(spectrum.real**2 + spectrum.imag**2)
    return magnitude

def get_phase(spectrum):
    return np.angle(spectrum)

def to_logscale(img):
    return np.log(1+img)

