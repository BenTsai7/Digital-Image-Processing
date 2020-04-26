import cv2
import numpy as np
import math

def make_smooth_filter(size):
    return [[1/(size*size)] * size for i in range(size)]

def make_laplacian_filter():
    filter = [[-1] * 3 for i in range(0,3)]
    filter[1][1] = 8
    return filter

def filter2d(img,filter):
    new_img = img.copy()
    center = len(filter)//2
    for x in range(1,img.shape[1]-1):
        for y in range(1,img.shape[0]-1):
            sum = 0
            for j in range(0,len(filter)):
                for k in range(0,len(filter)):
                    x_offset = center-j
                    y_offset = center-k
                    x_pos = x + x_offset
                    y_pos = y + y_offset
                    if(x_pos>=0 and x_pos<img.shape[1] and y_pos>=0 and y_pos<img.shape[0]):
                        sum += (filter[j][k] * img[y_pos][x_pos])
            new_img[y][x] = sum
    return new_img

def filter_laplacian(img,filter):
    new_img = img.copy()
    center = len(filter)//2
    for x in range(1,img.shape[1]-1):
        for y in range(1,img.shape[0]-1):
            sum = 0
            for j in range(0,len(filter)):
                for k in range(0,len(filter)):
                    x_offset = center-j
                    y_offset = center-k
                    x_pos = x + x_offset
                    y_pos = y + y_offset
                    if(x_pos>=0 and x_pos<img.shape[1] and y_pos>=0 and y_pos<img.shape[0]):
                        sum += (filter[j][k] * img[y_pos][x_pos])
            #overflow process
            if(sum<0):
                sum = 0
            elif(sum>255):
                sum = 255
            new_img[y][x] += sum
    return new_img


def high_boost_filter(img):
    k = 1.5
    filter = make_smooth_filter(3)
    new_img = filter2d(img,filter)
    high_boost_img = img.copy()
    for x in range(0,img.shape[1]):
        for y in range(0,img.shape[0]):
            mask = new_img[y][x] - img[y][x]
            g = high_boost_img[y][x] + k * mask
            if(g<0):
                g = 0
            elif(g>255):
                g = 255
            high_boost_img[y][x] = g
    return high_boost_img

def filter_main(img):
    #smooth image using filter with diffrent sizes
    filter = make_smooth_filter(3)
    new_img = filter2d(img,filter)
    cv2.imshow('Image', new_img)
    cv2.waitKey()
    filter = make_smooth_filter(5)
    new_img = filter2d(img,filter)
    cv2.imshow('Image', new_img)
    cv2.waitKey()
    filter = make_smooth_filter(7)
    new_img = filter2d(img,filter)
    cv2.imshow('Image', new_img)
    cv2.waitKey()
    #laplacian
    filter = make_laplacian_filter()
    new_img = filter_laplacian(img,filter)
    cv2.imshow('Image', new_img)
    cv2.waitKey()
    #high boost
    new_img = high_boost_filter(img)
    cv2.imshow('Image', new_img)
    cv2.waitKey()
    

raw_img = cv2.imread('xx.png')
img = cv2.cvtColor(raw_img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Image',img)
cv2.waitKey()
filter_main(img)

