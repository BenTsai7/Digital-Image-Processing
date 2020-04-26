import cv2
import numpy as np
import math

def centralization(img):
    #对图像进行中心化处理
    #即乘以(-1)的x+y次方
    M, N = img.shape
    for x in range(M):
        for y in range(N):
            img[x][y] = ((-1)**(x+y))*img[x][y]


def uncentralization(img):
    #对图像进行去中心化处理，用于逆变换
    #即乘以(-1)的x+y次方，使得图像的值变换回去
    M, N = img.shape
    for x in range(M):
        for y in range(N):
            img[x][y] = ((-1)**(x+y))*img[x][y]

def dft2d(img,flag):
    #DFT
    if(flag):
       return dft(img)
    #IDFT
    else:
       return idft(img)


def dft1d(vector):
    #1维DFT
    #求出输入向量的长度
    M = vector.shape[0]
    V = np.array([[np.exp(-1j*2*np.pi*u*x/M) for x in range(M)] for u in range(M)])
    return V.dot(vector) #点乘相当于为每一项指数项乘以对应的f(x,y)

def idft1d(vector):
    #1维IDFT
    #与DFT基本一致,只不过求指数时指数的虚部为负数，同时最后要除以一个系数长度
    M = vector.shape[0]
    V = np.array([[np.exp(1j*2*np.pi*u*x/M) for x in range(M)] for u in range(M)])
    return V.dot(vector) / M

def dft(img):
    #获取宽度和长度   
    M,N = img.shape
    #中心化处理
    centralization(img)
    #用complext复数来存取傅里叶图像的实部和虚部
    dft_img = np.zeros((M, N), dtype="complex128")
    #由傅里叶变换的可分性对两个X,Y维度分别做二个1维DFT
    for x in range(M):
        dft_img[x,:] = dft1d(img[x,:])
    for y in range(N):
        dft_img[:,y] = dft1d(dft_img[:,y])
    return dft_img

def idft(img):
    #获取长度和宽度   
    M,N = img.shape
    #用complext复数来存取傅里叶图像的实部和虚部
    idft_img = np.zeros((M, N), dtype="complex128")
    #实际上过程和dft基本一样
    for x in range(M):
        idft_img[x,:] = idft1d(img[x,:])
    for y in range(N):
        idft_img[:,y] = idft1d(idft_img[:,y]) 
    #舍弃掉虚部，返回实部且整数化后的图像
    #去中心化
    idft_int_img = np.array(idft_img.real.astype(int),dtype=np.uint8)
    uncentralization(idft_int_img)
    return idft_int_img


def show_fourier_img(img):
    #获取长度和宽度   
    M,N = img.shape
    fourier_img = np.zeros((M, N), dtype="float")
    #求复数模即幅值
    for x in range(M):
       for y in range(N):
           fourier_img[x][y] = np.sqrt(abs(img[x][y]))
    #标定到0-255才能显示
    #求出每个级别灰度的范围
    gray_level_offset = (fourier_img.max()-fourier_img.min())/256
    fourier_img[x][y] = int(fourier_img[x][y]/gray_level_offset)
    fourier_int_img = np.zeros((M, N), dtype="uint8")
    for x in range(M):
       for y in range(N):
           fourier_int_img[x][y] = int(fourier_img[x][y])
    cv2.imshow('Fourier Image',fourier_int_img)
    cv2.waitKey()

    

def dft_idft_process(img):
    #DFT
    dft_img = dft2d(img,True)
    #显示傅里叶变换后的图像
    show_fourier_img(dft_img)
    #IDFT
    idft_img = dft2d(dft_img,False)
    #显示逆变换后的图像
    cv2.imshow('IDFT Image',idft_img)
    cv2.waitKey()


def fft_ifft_process(img):
    #FFT
    fft_img = fft2d(img,True)
    #显示傅里叶变换后的图像
    show_fourier_img(fft_img)
    #IFFT
    ifft_img = fft2d(fft_img,False)
    #显示逆变换后的图像
    cv2.imshow('IFFT Image',ifft_img)
    cv2.waitKey()

def fft2d(img,flag):
    #DFT
    if(flag):
       return fft(img)
    #IDFT
    else:
       return ifft(img)

def fft(img):
    #获取宽度和长度   
    M,N = img.shape
    #中心化处理
    centralization(img)
    #用complext复数来存取傅里叶图像的实部和虚部
    fft_img = np.zeros((M, N), dtype="complex128")
    #由傅里叶变换的可分性对两个X,Y维度分别做二个1维FFT
    for x in range(M):
        fft_img[x,:] = fft1d(img[x,:])
    for y in range(N):
        fft_img[:,y] = fft1d(fft_img[:,y])
    return fft_img

def ifft(img):
    #获取长度和宽度   
    M,N = img.shape
    #用complext复数来存取傅里叶图像的实部和虚部
    ifft_img = np.zeros((M, N), dtype="complex128")
    #实际上过程和fft基本一样,除了要求conjugate以外
    img = img.conjugate()
    for x in range(M):
        ifft_img[x,:] = ifft1d(img[x,:])
    for y in range(N):
        ifft_img[:,y] = ifft1d(ifft_img[:,y]) 
    #舍弃掉虚部，返回实部且整数化后的图像
    #去中心化
    ifft_int_img = np.array(ifft_img.conjugate().real.astype(int)/(M*N),dtype=np.uint8)
    uncentralization(ifft_int_img)
    return ifft_int_img

def fft1d(vector):
    #1维FFT使用分治法
    #求出输入向量的长度
    M = vector.shape[0]
    if M%2 != 0:  #FFT必须为偶数长度,这里没有使用0-padding来使得fft的向量长度变为偶数
        print("Error: FFT vector not even")
        return
    if M<9: #由于没有使用0-padding来使得fft的向量长度变为偶数，M足够小则调用DFT进行计算
        return dft1d(vector)
    #递归调用1 dimension FFT
    even = fft1d(vector[::2])#利用python数组的切片，切出从0开始步长为2的偶数序列，然后递归调用求出
    odd = fft1d(vector[1::2])#利用python数组的切片，切出从1开始步长为2的奇数序列,然后递归调用求出
    coefficient = np.exp(-1j * 2 * np.pi * np.arange(M) / M)#实际上求得是odd和even的共同系数W_M——ux即W_2k_ux
    #concatenate分别求前半部分和后半部然后进行连接
    return np.concatenate([even + coefficient[:int(M/2)] * odd,even + coefficient[int(M/2):]*odd])

def ifft1d(vector):
    #1维IFFT
    #与FFT一致
    M = vector.shape[0]
    if M%2 != 0: #FFT必须为偶数长度,这里没有使用0-padding来使得fft的向量长度变为偶数
        print("Error: FFT vector not even")
        return
    if M < 9: #由于没有使用0-padding来使得fft的向量长度变为偶数，M足够小则调用DFT进行计算
        return dft1d(vector)
    #递归调用1 dimension FFT
    even = fft1d(vector[::2])#利用python数组的切片，切出从0开始步长为2的偶数序列
    odd = fft1d(vector[1::2])#利用python数组的切片，切出从1开始步长为2的奇数序列
    coefficient = np.exp(-1j * 2 * np.pi * np.arange(M) / M)#实际上求得是odd和even的共同系数
     #concatenate分别求前半部分和后半部然后进行连接
    return np.concatenate([even + coefficient[:int(M/2)] * odd,even + coefficient[int(M/2):]*odd])

def padding(img,P,Q):
     M,N = img.shape
     padding_img = np.zeros((P, Q), dtype="int")
     for i in range(P):
         for j in range(Q):
             if(i<M and j<N):
                padding_img[i][j]=img[i][j]
             else:
                padding_img[i][j]=0
     return padding_img

def padding_filter(filter,P,Q):
    M,N = filter.shape
    new_filter = np.zeros((P, Q), dtype="float")
    for i in range(P):
        for j in range(Q):
            if(i<M and j<N):
               new_filter[i][j]=filter[i][j]
            else:
               new_filter[i][j]=0
    return new_filter

def filter2d_freq(img,filter):
    M,N = img.shape
    a,b = filter.shape
    padding_img = padding(img,M+a-1,N+b-1) #扩充到防止干扰的最小范围
    filter = padding_filter(filter,M+a-1,N+b-1)
    centralization(filter)
    centralization(padding_img)
    #由于标准库的FFT计算速度快，这里直接调用标准库的接口
    #FFT
    fft_img = np.fft.fft2(padding_img)
    fft_filter = np.fft.fft2(filter)
    #IFFT
    ifft_img = np.fft.ifft2(fft_filter*fft_img)
    uncentralization(ifft_img)
    #显示滤波后的图像
    filter_img = np.zeros((M, N), dtype="int")
    #标定到0-255才能显示
    #求出每个级别灰度的范围
    gray_level_offset = (ifft_img.max()-ifft_img.min())/256
    for x in range(M):
       for y in range(N):
           filter_img[x][y] = int(np.real(ifft_img[x][y])/gray_level_offset)
    return filter_img


def filtering_process(img):
    filter = np.ones((5, 5), dtype="float") / 25 #5X5均值滤波器
    filter_img = filter2d_freq(img.copy(),filter)
    display_img = np.zeros((filter_img.shape[0], filter_img.shape[1]), dtype="uint8")
    for x in range(filter_img.shape[0]):
       for y in range(filter_img.shape[1]):
           display_img[x][y] = filter_img[x][y]
    cv2.imshow('Smooth Image',display_img)
    cv2.waitKey()
    
    #拉普拉斯滤波
    filter = np.zeros((3, 3), dtype="float") #3X3拉普拉斯滤波器
    filter[1][1]=-4
    filter[0][1]=1
    filter[1][0]=1
    filter[1][2]=1
    filter[2][1]=1
    filter_img = filter2d_freq(img.copy(),filter)
    display_img = np.zeros((filter_img.shape[0], filter_img.shape[1]), dtype="uint8")
    for x in range(filter_img.shape[0]):
       for y in range(filter_img.shape[1]):
           sum = img[x][y] + filter_img[x][y] #拉普拉斯变换还要叠加原图像
           #防止溢出
           if(sum>255):
               sum=255
           if(sum<0):
               sum = 0
           display_img[x][y] = sum 
    cv2.imshow('Laplacian Image',display_img)
    cv2.waitKey()


raw_img = cv2.imread('xx.png')
img = cv2.cvtColor(raw_img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Image',img)
cv2.waitKey()
#dft_idft_process(img) #DFT 耗时约为3-5分钟
img = cv2.cvtColor(raw_img,cv2.COLOR_BGR2GRAY)
fft_ifft_process(img) #FFT and IFFT
img = cv2.cvtColor(raw_img,cv2.COLOR_BGR2GRAY)
filtering_process(img)