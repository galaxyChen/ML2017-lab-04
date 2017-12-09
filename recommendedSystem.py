from sklearn import datasets
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt# -*- coding: utf-8 -*-

def loss(x,theta,y,r,lamb):
    temp = x*theta.T-y
    J = 0.5*((temp.A*r.A)**2).sum()+lamb/2*(theta.A**2).sum()+lamb/2*(x.A**2).sum()
    return J

def gradient(x,theta,y,r,lamb):
    temp = (x*theta.T-y).A*r.A
    grad_x = temp*theta + lamb*x
    grad_theta = temp.T*x + lamb*theta
    return grad_x,grad_theta

def gradientDecent(x,theta,y,r,lamb,alpha,num_of_round):
    print("gradient decent")
    loss_history = []
    loss_history.append(loss(x,theta,y,r,lamb))
    
    for i in range(num_of_round):
        grad_x,grad_theta = gradient(x,theta,y,r,lamb)
        x = x - alpha*grad_x
        theta = theta - alpha*grad_theta
        l = loss(x,theta,y,r,lamb)
        print("the %d round,loss: %f"%(i,l))
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
    lamb = 1
    #learning rate
    alpha = 0.1
    num_of_round = 100
    x,theta,loss_history=gradientDecent(init_x,init_theta,y,r,lamb,alpha,num_of_round)
    plt.plot(np.arange(num_of_round+1),loss_history,label='loss')
    plt.legend(loc=1)
    plt.xlabel('number_of_rounds')
    plt.ylabel('loss')
    return x,theta
    
def predict(x,theta):
    return x*theta.T


def getData():
    train_y = []
    train_r = []
    test_y = []
    test_r = []
    return train_y,train_r,test_y,test_r


