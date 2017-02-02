#coding=utf-8
#!/usr/bin/env python

import random

class Connection(object):
    def __init__(self,upstream_node,downstream_node):
        '''
        初始化连接，权重初始化为一个很小的随机值
        :param upstream_node: 上游节点
        :param downstream_node: 下游节点
        '''
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1,0.1)
        self.gradient = 0.0

    def calc_gradient(self):
        '''
        计算梯度
        :return:
        '''
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def get_gradient(self):
        return self.gradient

    def update_weight(self,rate):
        '''
        根据梯度下降算法更新权重
        :param rate:
        :return:
        '''
        self.calc_gradient()
        self.weight += rate * self.gradient

    def __str__(self):
        '''
        打印连接信息
        :return:
        '''
        return '(%u-%u)->(%u-%u) = %f' %(
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight )

class Connections(object):
    def __init__(self):
        self.connections = []

    def add_connection(self,connection):
        self.connections.append(connection)

    def dump(self):
        for conn in self.connections:
            print conn
