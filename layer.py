#coding=utf-8
#!/usr/bin/env python

from node import *

class Layer(object):
    def __index__(self,layer_index,node_count):
        '''
        初始化一层
        :param layer_index: 层id
        :param node_count: 包含的节点数
        :return:
        '''
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index,i))
        self.nodes.append(ConstNode(layer_index,node_count))

    def set_output(self,data):
        '''
        设置输入层的数据
        :param data:
        :return:
        '''
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        '''
        非输入层计算输出
        :return:
        '''

        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        '''
        打印层信息
        :return:
        '''
        for node in self.nodes:
            print node



