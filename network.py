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