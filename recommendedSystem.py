# write your from sklearn import datasets
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt# -*- coding: utf-8 -*-
from collections import defaultdict
import pandas as pd
# 计算loss
def loss(x,theta,y,r):
    n = r.sum()
    temp = (x*theta.T-y).A*r.A
    J = 0.5*(temp**2).sum()/n
    return J
# 计算梯度
def gradient(x,theta,y,r,lamb):
    # 参考代码说明中的梯度计算公式
    # lamb是正则化项
    temp = (x*theta.T-y).A*r.A
    mx = x.shape[0]
    mtheta = theta.shape[0]
    grad_x = temp*theta + lamb*x
    grad_theta = temp.T*x + lamb*theta
    return grad_x/mx,grad_theta/mtheta

# 随机梯度下降
def SGD(x,theta,y,r,lamb,alpha,num_of_round):
    print("stochastic gradient decent")
    # 保存还没有训练的时候的初始loss
    loss_history = []
    loss_history.append(loss(x,theta,y,r))
    # nu是用户数量，nm是电影数量
    nu = theta.shape[0]
    nm = x.shape[0]
    
    mx = x.shape[0]
    mtheta = theta.shape[0]
    
    for index in range(num_of_round):
        # 随机选取一行一列（一个用户和一个电影）
        rand_i = np.random.randint(0,nm)
        rand_j = np.random.randint(0,nu)
        # 取出电影向量
        x_i = x[rand_i,:]
        # 计算这个电影向量的梯度
        grad_x = ((x_i*theta.T-y[rand_i,:]).A*r[rand_i,:].A)*theta
        # 取出用户向量
        theta_j = theta[rand_j,:]
        # 计算这个用户向量的梯度
        grad_theta = ((theta_j*x.T-y[:,rand_j].T).A*r[:,rand_j].T.A)*x
        # 利用梯度更新随机选取的样本的参数
        x[rand_i,:] = x[rand_i,:] - alpha*grad_x/mx
        theta[rand_j,:] = theta[rand_j,:] - alpha*grad_theta/mtheta
        # 计算这一次迭代的loss值
        l = loss(x,theta,y,r)
        if index%10000==0:
            print("the %d round,loss: %f"%(index,l))
        loss_history.append(l)
        
    return x,theta,loss_history

# 批量梯度下降（使用所有的样本）
def gradientDecent(x,theta,y,r,lamb,alpha,num_of_round):
    print("batch gradient decent")
    loss_history = []
    loss_history.append(loss(x,theta,y,r))
    
    for i in range(num_of_round):
        # 计算梯度
        grad_x,grad_theta = gradient(x,theta,y,r,lamb)
        # 更新参数
        x = x - alpha*grad_x
        theta = theta - alpha*grad_theta
        # 计算loss
        l = loss(x,theta,y,r)
        if i%100==0:
            print("the %d round,loss: %f"%(i+1,l))
        loss_history.append(l)
        
    return x,theta,loss_history
    
# 训练函数  
def train(y,r,k):
    print("train begin")
    nm = y.shape[0]
    nu = y.shape[1]
    #初始化电影矩阵为随机的0~1
    init_x = np.matrix(np.random.rand(nm,k))
    #初始化用户矩阵为随机的0~1
    init_theta = np.matrix(np.random.rand(nu,k))
    #正则化项
    lamb = 10
    #学习率
    alpha = 0.1
    # 循环次数，在这里批量下降可以选300，随机梯度下降可以选30000
    num_of_round_bgd = 300
    num_of_round_sgd = 300
    # 分别训练
    x,theta,bgd_loss_history=gradientDecent(init_x,init_theta,y,r,lamb,alpha,num_of_round_bgd)
    x,theta,sgd_loss_history=SGD(init_x,init_theta,y,r,lamb,alpha,num_of_round_sgd)
    # 打印图
    plt.plot(np.arange(num_of_round_bgd+1),bgd_loss_history,label='bgd loss')
    plt.legend(loc=1)
    plt.xlabel('number_of_rounds')
    plt.ylabel('loss')
    plt.figure()
    plt.plot(np.arange(num_of_round_sgd+1),sgd_loss_history,label='sgd loss')
    plt.legend(loc=1)
    plt.xlabel('number_of_rounds')
    plt.ylabel('loss')
    return x,theta
    
def predict(x,theta):
    # 转置相乘就是预测结果
    return x*theta.T


def getData():
    print("generate data")
    data = pd.read_table('./ml_100k/u1.base',names=['user_id','item_id','rating','timestamp'])[['user_id','item_id','rating']]
    data2 = pd.read_table('./ml_100k/u1.test',names=['user_id','item_id','rating','timestamp'])[['user_id','item_id','rating']]
    train_data = data.pivot_table(values='rating',index='item_id',columns='user_id')
    train_r_data = (~train_data.isnull()).astype(int)
    train_set = np.matrix(train_data.fillna(0).as_matrix())
    train_r = np.matrix(train_r_data.as_matrix())
        
    item_set = set(data['item_id'].values)
    user_set = set(data['user_id'].values)
    
    t = data2.apply(lambda x:(x['user_id'] in user_set) and (x['item_id'] in item_set),axis=1)
    data2 = data2[t]
    
    
    test_data = train_data.to_dict()
    test_r = train_r_data.to_dict()
    for user in test_data:
        for item in test_data[user]:
            test_data[user][item]=0
            test_r[user][item]=0
            
            
    for i in range(len(data2)):
        row = data2.iloc[i,:]
        test_data[row['user_id']][row['item_id']] = row['rating']
        test_r[row['user_id']][row['item_id']] = 1
    
    test_data = pd.DataFrame(test_data)
    test_r = pd.DataFrame(test_r)
    test_set = np.matrix(test_data.fillna(0).as_matrix())
    test_r = np.matrix(test_r.as_matrix())
    
    return train_set,train_r,test_set,test_r

train_set,train_r,test_set,test_r = getData()
k = 100
x,theta = train(train_set,train_r,k)
print("")
print("test loss is %f"%loss(x,theta,test_set,test_r))

