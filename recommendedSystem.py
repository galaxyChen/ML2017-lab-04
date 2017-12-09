from sklearn import datasets
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt# -*- coding: utf-8 -*-
from collections import defaultdict
import pandas as pd

def loss(x,theta,y,r):
    n = r.sum()
    temp = (x*theta.T-y).A*r.A
    J = 0.5*(temp**2).sum()/n
    return J

def gradient(x,theta,y,r,lamb):
    temp = (x*theta.T-y).A*r.A
    mx = x.shape[0]
    mtheta = theta.shape[0]
    grad_x = temp*theta + lamb*x
    grad_theta = temp.T*x + lamb*theta
    return grad_x/mx,grad_theta/mtheta

def RGD(x,theta,y,r,lamb,alpha,num_of_round):
    print("gradient decent")
    loss_history = []
    loss_history.append(loss(x,theta,y,r))
    nu = theta.shape[0]
    nm = x.shape[0]
    
    mx = x.shape[0]
    mtheta = theta.shape[0]
    
    for index in range(num_of_round):
        rand_i = np.random.randint(0,nm)
        rand_j = np.random.randint(0,nu)
        x_i = x[rand_i,:]
        grad_x = ((x_i*theta.T-y[rand_i,:]).A*r[rand_i,:].A)*theta
        theta_j = theta[rand_j,:]
        grad_theta = ((theta_j*x.T-y[:,rand_j].T).A*r[:,rand_j].T.A)*x
        
        x[rand_i,:] = x[rand_i,:] - alpha*grad_x/mx
        theta[rand_j,:] = theta[rand_j,:] - alpha*grad_theta/mtheta
        
        l = loss(x,theta,y,r)
        if index%1000==0:
            print("the %d round,loss: %f"%(index,l))
        loss_history.append(l)
        
    return x,theta,loss_history

def gradientDecent(x,theta,y,r,lamb,alpha,num_of_round):
    print("gradient decent")
    loss_history = []
    loss_history.append(loss(x,theta,y,r))
    
    for i in range(num_of_round):
        grad_x,grad_theta = gradient(x,theta,y,r,lamb)
        x = x - alpha*grad_x
        theta = theta - alpha*grad_theta
        l = loss(x,theta,y,r)
        print("the %d round,loss: %f"%(i+1,l))
        loss_history.append(l)
        
    return x,theta,loss_history
    
    
def train(y,r,k):
    print("train begin")
    nm = y.shape[0]
    nu = y.shape[1]
    #movie matrix
    init_x = np.matrix(np.random.rand(nm,k))
    #user matrix
    init_theta = np.matrix(np.random.rand(nu,k))
    #regularization term
    lamb = 10
    #learning rate
    alpha = 0.1
    num_of_round = 30000
    x,theta,loss_history=RGD(init_x,init_theta,y,r,lamb,alpha,num_of_round)
    plt.plot(np.arange(num_of_round+1),loss_history,label='loss')
    plt.legend(loc=1)
    plt.xlabel('number_of_rounds')
    plt.ylabel('loss')
    return x,theta
    
def predict(x,theta):
    return x*theta.T


def getData():
    data = pd.read_table('./ml_100k/u1.base',names=['user_id','item_id','rating','timestamp'])[['user_id','item_id','rating']]
    data2 = pd.read_table('./ml_100k/u1.test',names=['user_id','item_id','rating','timestamp'])[['user_id','item_id','rating']]
    train = defaultdict(lambda : defaultdict(lambda : 0))
    train_r = defaultdict(lambda : defaultdict(lambda : 0))
    print("generating training data")
    for i in range(len(data)):
        row = data.iloc[i,:]
        train[row['item_id']][row['user_id']]=row['rating']
        train_r[row['item_id']][row['user_id']]=1
        
    
    train_data = pd.DataFrame(train).fillna(0)
    train_data_r = pd.DataFrame(train_r).fillna(0)
    train_r_m = np.matrix(train_data_r.as_matrix())
    train_data_m = np.matrix(train_data.as_matrix())
    
    print("generating test data")
    test = train.copy()
    test_r = train_r.copy()
    for x in test:
        for y in test[x]:
            test[x][y]=np.nan
            test_r[x][y]=0
    
    for i in range(len(data2)):
        row = data.iloc[i,:]
        test[row['item_id']][row['user_id']]=row['rating']
        test_r[row['item_id']][row['user_id']]=1
        
    
    test_data = pd.DataFrame(test).fillna(0)
    test_data_r = pd.DataFrame(test_r).fillna(0)
    test_r_m = np.matrix(test_data_r.as_matrix())
    test_data_m = np.matrix(test_data.as_matrix())


    print("generating done")
    return train_data_m,train_r_m,test_data_m,test_r_m


train_set,train_r,test_set,test_r = getData()
k = 100
x,theta = train(train_set,train_r,k)
print("")
print("test loss is %f"%loss(x,theta,test_set,test_r))


