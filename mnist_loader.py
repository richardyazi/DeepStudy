#coding=utf-8
#!/usr/bin/env python

import struct

#数据加载基类
class Loader(object):
    def __init__(self,path,count):
        '''
        初始化加载器
        :param path:
        :param count:
        '''
        self.path = path
        self.count = count

    def get_file_content(self):
        '''
        读取文件内容
        :return:
        '''
        f = open(self.path,'rb')
        content = f.read()
        f.close()
        return content

    def to_int(self,byte):
        '''
        将unsigned byte字符转换为整数
        :param byte:
        :return:
        '''
        return struct.unpack('B',byte)[0]

class ImageLoader(Loader):
    def get_picture(self,content,index):
        '''
        内部函数，从文件中获取图像
        :param content:
        :param index:
        :return:
        '''
        start = index*28*28+16
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(self.to_int(content[start+i * 28 +j]))
        return picture

    def get_one_sample(self,picture):
        '''
        内部函数，将图像转换为样本的输入向量
        :param picture:
        :return:
        '''
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    def load(self):
        '''
        加载数据文件，获得全体样本的输入向量
        :return:
        '''
        content = self.get_file_content()
        data_set = []
        for index  in range(self.count):
            data_set.append(self.get_one_sample(self.get_picture(content,index)))
        return data_set

class LabelLoader(Loader):
    def load(self):
        '''
        加载数据文件，获得全体样本向量
        :return:
        '''
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index+8]))
        return labels

    def norm(self,label):
        '''
        内部函数，将一个值转换称10维标签向量
        :param label:
        :return:
        '''
        label_vec = []
        label_value = self.to_int(label)
        for i in range(10):
            if i == label_value:
                label_vec.append(0.95)
            else:
                label_vec.append(0.05)
        return label_vec