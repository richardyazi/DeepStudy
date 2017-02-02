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







