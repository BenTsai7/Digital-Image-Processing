import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def compute_histogram(img):
    histogram = np.zeros(256)
    for x in range(0,img.shape[1]):
        for y in range(0,img.shape[0]):
            histogram[img[y][x]]+=1
    return histogram


def show_histogram(img):
    #use hist to draw histogram
    plt.hist(img.ravel(),bins=256)
    plt.show()


def equalize(img,raw_histogram):
    #compute transform relations
    transform_table = np.zeros(256)
    for i in range(0,len(transform_table)):
        sum = 0
        for j in range(0,i+1):
            sum += raw_histogram[j]
        sum = sum * 255 / (img.shape[0]*img.shape[1])
        transform_table[i]= int(round(sum))
    #equalize image
    for x in range(0,img.shape[1]):
        for y in range(0,img.shape[0]):
            img[y][x]= transform_table[int(img[y][x])]


def equalize_hist(img):
    raw_histogram = compute_histogram(img)
    show_histogram(img)
    equalize(img,raw_histogram)
    cv2.imshow('Image',img)
    cv2.waitKey()
    equal_histogram1 = compute_histogram(img)
    show_histogram(img)
    equalize(img,equal_histogram1)
    cv2.imshow('Image',img)
    cv2.waitKey()
    show_histogram(img)


raw_img = cv2.imread('xx.png')
img = cv2.cvtColor(raw_img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Image',img)
cv2.waitKey()
equalize_hist(img)