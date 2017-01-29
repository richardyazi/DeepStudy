#!/usr/bin/env python

class Preceptron(object):
    def __init__(self,input_num,activator):
        self.activator = activator
        self.weights = {0.0 for _ in range(input_num)}
        self.bias = 0.0

    def __str__(self):
        return 'weights\t:%s\n\bias\t:%f\n' % (self.weights,self.bias)

    def predict(self,input_vec):
        '''
        输入向量，输出感知器结果
        '''
        return self.activator(
            reduce(lambda a,b:a+b,
                   map(lambda a,b:a*b,
                       zip(self.weights,input_vec),
                       self.bias)))

    def train(self,input_vecs,labels,iteration,rate):
        '''
        数据训练数据:一组向量，对应的labels，训练轮数，学习率
        '''
        for i in range(iteration):
            self._one_iteration(input_vecs,labels,rate)

    def _one_iteration(self,input_vecs,labels,rate):
        '''
        一次迭代，把所有的训练数据过一遍
        '''
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        # 而每个训练样本是(input_vec, label)
        samples = zip(input_vecs,labels)
        for (input_vec,label) in samples:
            output = self.predict(input_vec)
            self._update_weights(input_vec,output,label,rate)

    def _update_weights(self,input_vec,output,label,rate):
        '''
        按照感知器规则更新权重
        '''
        delta = label - output
        self.weights = map(lambda (x,w) : w + rate* delta*x,zip(input_vec,self.weights))
        self.bias += rate* delta


