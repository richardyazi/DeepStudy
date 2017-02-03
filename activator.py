#coding=utf-8
#!/usr/bin/env python

import numpy as np

class ReluActivator(object):
    '''
    Relu激活函数
    '''
    def forward(self,weight_input):
        '''
        激活函数
        :param weight_input: 加权输入
        :return:
        '''
        return max(0,weight_input)

    def backward(self,output):
        '''
        导数
        :param output:
        :return:
        '''
        return 1 if output >0 else 0

class IdentityActivator(object):
    def forward(self,weight_input):
        '''
        激活函数 f(x)= x
        :param weight_input: 加权输入
        :return:
        '''
        return weight_input

    def backward(self,output):
        '''
        导数
        :param output:
        :return:
        '''
        return 1