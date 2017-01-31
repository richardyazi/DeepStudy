#coding=utf-8
#!/usr/bin/env python

from connection import *
from layer import *

class Network(object):
    def __init__(self,layers):
        '''
        全连接神经网络初始化
        :param layers: 二维数组，定义每层节点数
        '''
        self.connections =  Connections()
        self.layers =[]
        layer_count = len(layers)
        node_count = 0
        for i in range(layer_count):
            self.layers.append(Layer(i,layers[i]))
        for layer in range(layer_count-1):
            connections = [
                Connection(upstream_node,downstram_node)
                for upstream_node in self.layers[layer].nodes
                for downstram_node in self.layers[layer+1].nodes ]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    def train(self,labels,data_set,rate,iteration):
        '''
        训练神经网络
        :param labels: 训练样本标签
        :param data_set: 二位数组，训练样本特性
        :param rate:
        :param iteration:
        :return:
        '''
        for i in range(iteration):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d],data_set[d],rate)

    def train_one_sample(self,label,sample,rate):
        '''
        内部函数，用一个样本训练网络
        :param label:
        :param sample:
        :param rate:
        :return:
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def predict(self,sample):
        '''
        根据输入计算样本预期输出值
        :param sample:
        :return:
        '''
        self.layers[0].set_output(sample)
        for i in range(1,len(self.layers)):
            self.layers[i].calc_output()

    def calc_delta(self,label):
        '''
        内部函数，计算每个节点的dalta
        :param label:
        :return:
        '''
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self,rate):
        '''
        内部函数，更新权重
        :param rate:
        :return:
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def clac_gradient(self):
        '''
        内部函数，计算每个连接的梯度
        :return:
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.clac_gradient()

    def get_gradient(self,label,sample):
        '''
        获得网络在一个样本下的各连接梯度
        :param label: 标签
        :param sample: 样本
        :return:
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.clac_gradient()

    def dump(self):
        '''
        打印神经网络的基本信息
        :return:
        '''
        for layer in self.layers:
            layer.dump()