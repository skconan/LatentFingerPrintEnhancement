import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

def get_file_path(dir_name):
    """
      Get all files in directory "dir_name"
    """
    file_list = os.listdir(dir_name)
    files = []
    for f in file_list:
        abs_path = os.path.join(dir_name, f)
        if os.path.isdir(abs_path):
            files = files + get_file_path(abs_path)
        else:
            files.append(abs_path)
    return files


def get_file_name(img_path):
    if "\\" in img_path:
        name = img_path.split('\\')[-1]
    else:
        name = img_path.split('/')[-1]

    name = name.replace('.gif', '')
    name = name.replace('.png', '')
    name = name.replace('.jpg', '')
    return name


def get_kernel(shape='rect', ksize=(3, 3)):
    if shape == 'rect':
        return cv.getStructuringElement(cv.MORPH_RECT, ksize)
    elif shape == 'ellipse':
        return cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize)
    elif shape == 'plus':
        return cv.getStructuringElement(cv.MORPH_CROSS, ksize)
    else:
        return None


def apply_clahe(img_bgr):
    lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2Lab)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv.merge((l, a, b))
    res = cv.cvtColor(lab, cv.COLOR_Lab2BGR)
    return res


def normalize(image, max=255, input_max=None, input_min=None):
    # image = np.array(image,np.float32)
    print(image.max())
    print(image.min())
    if input_max is not None:
        result = 255.*(image - input_min)/(input_max-input_min)
    else:
        result = 255.*(image - image.min())/(image.max()-image.min())
    result = np.uint8(result)
    return result

def implot(figname, image):
    f= plt.figure(figname)
    plt.imshow(image)
    plt.show()
    cv.waitKey(-1)
    plt.close()    