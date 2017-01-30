#coding=utf-8
#!/usr/bin/env python

from perceptron import Perceptron

#定义激活函数
f = lambda x:x
class LinearUnit(Perceptron):
    def __init__(self,input_num):
        Perceptron.__init__(self,input_num,f)