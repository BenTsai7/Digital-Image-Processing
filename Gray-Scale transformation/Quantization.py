import cv2
import numpy
import math

def lower_bound(list, target): #二分搜索 lower_bound
    low, high = 0, len(list)-1
    pos = len(list)
    while low<high:
        mid = (low+high)//2
        if list[mid] < target:
            low = mid+1
        else:
            high = mid
    if list[low]>=target:
        pos = low
    return pos

def gray_level_quantization(img,level):
    gray_offset = math.floor(255/(level-1));
    newImg =numpy.zeros([img.shape[0],img.shape[1],1],dtype=numpy.uint8)
    gray_interval = []
    count = 0
    while count<256:
        gray_interval.append(count)
        count+=gray_offset
    for x in range(0,img.shape[1]):
        for y in range(0,img.shape[0]):
            pos = lower_bound(gray_interval,img[y][x][0])
            if pos==len(gray_interval):
                pos = len(gray_interval)-1
            if pos!=0:
                if img[y][x][0]-gray_interval[pos-1]<gray_interval[pos]-img[y][x][0]:
                    pos -= 1
            newImg[y][x] = gray_interval[pos]
    cv2.imshow('Image',newImg)
    cv2.waitKey()

img = cv2.imread('xx.png')
cv2.imshow('Image',img)
cv2.waitKey()
gray_level_quantization(img,128)
gray_level_quantization(img,64)
gray_level_quantization(img,32)
gray_level_quantization(img,8)
gray_level_quantization(img,2)