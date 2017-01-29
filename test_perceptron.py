#coding=utf-8
#!/usr/bin/env python
import perceptron

def f(x):
    '''
    定义激活函数f
    :param x: 加权输入
    :return:
    '''
    return 1 if x>0 else 0

def get_training_dataset():
    '''
    基于and的真值表构建测试数据
    :return:
    '''
    #输入向量列表
    input_vecs= [[1,0],[1,1],[0,0],[0,1]]
    #期望的输出列表，与输入一一对应
    labels = [0,1,0,0]
    return input_vecs,labels

def train_and_perceptron():
    '''
    使用and真值表训练感知器
    :return:
    '''
    #p = Perceptron.Perceptron(2,f)
    p = perceptron.Perceptron(2,f)
    input_vecs,labels = get_training_dataset()
    p.train(input_vecs,labels,10,0.1)
    return p

if __name__ == "__main__":
    #训练感知器
    and_perceptron = train_and_perceptron()
    #打印训练后的权重
    print and_perceptron

    #测试
    print '1 and 1 = %d'% and_perceptron.predict([1,1])
    print '1 and 0 = %d'% and_perceptron.predict([1,0])
    print '0 and 0 = %d'% and_perceptron.predict([0,0])
    print '0 and 1 = %d'% and_perceptron.predict([0,1])

