import cv2 as cv
import numpy as np
from utilities import *
from matplotlib import pyplot as plt

def spatial2freq(img):
    spectrum = np.fft.fft2(img)
    spectrum = np.fft.fftshift(spectrum)
    return spectrum


def freq2spatial(img):
    img = np.fft.ifftshift(img)
    img = np.fft.ifft2(img)
    mag = get_magnitude(img)
    mag = normalize(mag)
    return mag

def get_magnitude(spectrum):
    magnitude = np.abs(spectrum)
    return magnitude

def get_phase(spectrum):
    return np.angle(spectrum)

def to_logscale(img):
    return normalize(np.log(img))

