#coding=utf-8
#!/usr/bin/env python

import numpy as np

def padding(input_array,zp):
    '''
    为数组填充0
    :param input_array:
    :param zp:
    :return:
    '''

    if zp == 0 :
        return input_array
    else:
        if input_array.ndim == 3:
            input_depth = input_array.shape[0]
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            padded_array = np.zeros((
                input_depth,
                input_height + 2 * zp,
                input_width + 2 * zp))
            padded_array[:,
            zp:zp+input_height,
            zp:zp+input_width] = input_array
            return padded_array
        elif input_array.ndim == 2 :
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((input_height + 2 * zp,input_width + 2 * zp))
            padded_array[zp:zp+input_height,zp:zp+input_width] = input_array
            return padded_array

def conv(input_array,kernel_array,output_array,strike,bias):
    '''
    计算卷积，自动适配2D和3D
    :param input_array:输入数组
    :param kernel_array:卷积核 即权重集
    :param output_array:输出数组
    :param strike:步长
    :param bias:基本权重
    :return:
    '''
    #channel_number = input_array.ndim
    output_height,output_width = output_array.shape
    kernel_height,kernel_width = kernel_array[-2:]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (get_patch(input_array,i,kernel_width,kernel_height,
                                            strike)* kernel_array
                                 ).sum() + bias
    return output_array

def get_patch(input_array,idx_i,idx_j,kernel_width,kernel_height,strike):
    '''
    获得输出(i,j)对应的输入数组
    :param input_array:输入数组
    :param i: 输出对应的行号
    :param j: 输出对应的列号
    :param kernel_width: 卷积核的宽度
    :param kernel_height: 卷积核的长度c
    :param strike: 步长
    :return:
    '''
    start_i = idx_i * strike
    start_j = idx_j * strike

    if input_array.ndim == 3 :
        return input_array[:,
               start_j:start_j + kernel_height,
               start_i:start_i + kernel_width]
    elif input_array.ndim == 2:
        return input_array[start_j:start_j + kernel_height,
               start_i:start_i + kernel_width]



