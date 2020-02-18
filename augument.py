import os
import cv2
import numpy as np
from skimage import exposure, color, transform, util
import random

source = "data/"
destination = "data/"


def contrast_stretching(img):
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale

def flip(img):
    return np.fliplr(img)

def rotate(img):
    return transform.rotate(img, random.uniform(-20, 20), preserve_range=True)

def noise(img):
    img_nos = util.random_noise(img, mode='gaussian', clip = True)
    noise_img = np.array(255 * img_nos, dtype='uint8')
    return noise_img

def aug_function(img,i):
    mux = i
    # print(mux)
    if (mux==0):
        return contrast_stretching(img)
    elif (mux==1):
        return flip(img)
    elif (mux==2):
        return rotate(img)
    elif (mux==3):
        return noise(img)
    elif (mux==4):
        return rotate(contrast_stretching(noise(img)))
    elif (mux==5):
        return flip(rotate(contrast_stretching(img)))
    elif (mux==6):
        return rotate(noise(img))
    elif (mux==7):
        return flip(rotate(img))
    elif (mux==8):
        return flip(noise(img))
    elif (mux==9):
        return img

def offline_aug():
    for file in os.listdir(source):
        img_dir = os.path.join(source, file)
        img = cv2.imread(img_dir)
        print(file)
        cv2.imwrite(destination+"/"+file, img)
        for i in range(9):
            cv2.imwrite(destination + "/aug" + str(i) + file, aug_function(img, i))

if (__name__ == "__main__"):
    offline_aug()