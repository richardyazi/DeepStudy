#coding=utf-8
#!/usr/bin/env python

import numpy as np
from filter import *
from cnn import *

class ConvLayer(object):
    def __init__(self,
                 input_width,
                 input_height,
                 channel_number,
                 filter_width,
                 filter_height,
                 filter_number,
                 zero_padding,
                 strike,activator,learning_rate):
        '''
        构造一个卷积层
        :param input_width: 输入数据的宽度
        :param input_height:数据数据的长度
        :param channel_number:数据数据的深度
        :param filter_width:过滤器宽度
        :param filter_height:过滤器长度，深度和channel_number一样，所以不需要单独的定义
        :param filter_number:过滤器数量，影响输出的深度
        :param zero_padding:填零数量
        :param strike:步长
        :param activator: 激活函数
        :param learning_rate:学习率
        '''
        self.input_width =input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.strike = strike

        self.output_width = ConvLayer.calculate_output_size(input_width,filter_width,zero_padding,strike)
        self.output_height = ConvLayer.calculate_output_size(input_height,filter_height,zero_padding,strike)
        self.output_array = np.zeros(self.filter_number,self.output_height,self.output_width)
        self.filter_array = []
        for i in range(self.filter_number):
            self.filter_array.append(Filter(self.filter_width,self.filter_height,self.channel_number))
        self.activator = activator
        self.learning_rate = learning_rate

    @staticmethod
    def calculate_output_size(input_size,filter_size,zero_padding,strike):
        '''
        计算输出的纬度 (3维)
        :param filter_size:
        :param zero_padding:
        :param strike:
        :return:
        '''
        return (input_size - filter_size + 2 * zero_padding) / strike + 1

    def forward(self,input_array):
        '''
        计算卷积层结果并保存在self.output_array中
        :param input_array:
        :return:
        '''
        self.input_array = input_array

        #填充0
        self.padding_input_array = padding(self.input_array,self.zero_padding)
        for f in range(self.filter_number):
            filter = self.filter_array[f]
            conv(self.padding_input_array,filter.get_weight(),self.output_array[f],self.strike,filter.get_bias())
        element_wise_op(self.output_array,self.activator.forward)

    def bp_sensitivity_map(self,sensitivity_array,activator):
        '''
        计算传递到上一层的sentivity map
        :param sensitivity_array:
        :param activator: 激活函数
        :return:
        '''
        #处理卷积步长，对原始sensitivity map进行扩展
        #虽然补0也会产生残差，但是不需要往上传递
        expand_array = self.expand_sensitivity_map(sensitivity_array)

        #填充卷积使得sentivity map与weights的卷积计算的输出size与输入相等
        expand_height,expand_width = expand_array.shape[-2:]
        zp = (self.input_width - expand_width + self.filter_width - 1 ) /2
        padding_array = padding(expand_array,zp)

        #初始化delta_array用于保存到上一层的sentivity map
        self.delta_array = self.create_delta_array()

        #对于具有多个filter的卷积层，最终传到到上一层的sensitity map
        #是所有filter的sensitivy map之和
        for f in range(self.filter_number):
            filter = self.filter_array[f]
            #filter翻转180度
            flipped_weights = np.array(map(lambda i: np.rot90(i,2) , filter.get_weight))

            #计算与一个filter对应的delta_array
            delta_array = self.create_delta_array()
            for d in range(filter.shape[0]):
                conv(padding_array[f],flipped_weights[d],delta_array[d],1,0)
            self.delta_array += delta_array

            #将计算结果与激活函数的偏导数做element-wise乘法操作
            derivative_array = np.array(self.input_array)
            element_wise_op(derivative_array,self.activator.backward)
            self.delta_array *= derivative_array

    def expand_sensitivity_map(self,sensitivity_array):
        '''
        将sentivity array的步长降为1
        :param sensitivity_array:
        :return:
        '''
        depth = sensitivity_array.shape[0]
        #确定扩展后的sensitivy map的大小
        #计算strike为1时的sentivity map 的大小
        expand_width = self.input_width - self.filter_width + 2 * self.zero_padding + 1
        expand_height = self.input_height - self.filter_height + 2 * self.zero_padding +1

        #构建新的sentivity map
        expand_array = np.zeros((depth,expand_height,expand_width))

        #拷贝原始数据
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.strike
                j_pos = j * self.strike
                expand_array[:,i_pos,j_pos] = sensitivity_array[:,i,j]
        return expand_array

    def create_delta_array(self):
        return np.zeros((self.channel_number,self.input_height,self.input_width))

    def bp_gradint(self,sensitivity_map):
        #处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(sensitivity_map)

        for f in range(self.filter_number):
            #计算每个权重的梯度
            filter = self.filter_array[f]
            for d in range(filter.shape[0]):
                conv(self.padding_input_array[d],expanded_array,filter.weights_grad[d],1,0)
            #计算偏置项梯度
            filter.bias_grad = expanded_array[f].sum()










