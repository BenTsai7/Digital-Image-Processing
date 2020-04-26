import cv2
import numpy
import math
def bi_linear_interpolation(img, output_x, output_y):
    input_x = img.shape[1]
    input_y = img.shape[0]
    ratio_x = input_x/output_x;
    ratio_y = input_y/output_y;
    newImg =numpy.zeros([output_y,output_x,1],dtype=numpy.uint8)
    for x in range(0,output_x):
        for y in range(0,output_y):
            raw_x = (x+0.5) * ratio_x -0.5
            raw_y = (y+0.5) * ratio_y -0.5
            x1 = math.floor(raw_x)
            y1 = math.floor(raw_y)
            x2 = x1+1
            y2 = y1+1
            #原有的边界处理,防止溢出
            #if x1>=input_x:
            #   x1 = input_x-1
            #if y1>=input_y:
            #   y1=input_y-1
            #if x2>=input_x:
            #   x2= input_x-1
            #if y2>=input_y:
            #   y2=input_y-1
            #改进的边界处理
            if x1>=input_x or x2>=input_x:
                x1=input_x-2
                x2=input_x-1
            if y2>=input_y or y2>=input_y:
                y1=input_y-2;
                y2=input_y-1;

            #根据线性插值公式求解
            newImg[y][x]=img[y1][x1][0]*(x2-raw_x)*(y2-raw_y)+img[y2][x1][0]*(raw_x-x1)*(y2-raw_y)+img[y1][x2][0]*(x2-raw_x)*(raw_y-y1)+img[y2][x2][0]*(raw_x-x1)*(raw_y-y1)
    cv2.imshow('Image',newImg)
    cv2.waitKey()

img = cv2.imread('xx.png')
cv2.imshow('Image',img)
cv2.waitKey()
bi_linear_interpolation(img, 192, 168)
bi_linear_interpolation(img, 96, 64)
bi_linear_interpolation(img, 48, 32)
bi_linear_interpolation(img, 24, 16)
bi_linear_interpolation(img, 12, 8)
bi_linear_interpolation(img, 300, 200)
bi_linear_interpolation(img, 450, 300)
bi_linear_interpolation(img, 500, 200)