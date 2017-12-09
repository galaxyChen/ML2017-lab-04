# -*- coding: utf-8 -*-

import pandas as pd
from collections import defaultdict
import numpy as np

def loss(x,theta,y,r,lamb):
    temp = x*theta.T-y
    J = 0.5*((temp.A*r.A)**2).sum()+lamb/2*(theta.A**2).sum()+lamb/2*(x.A**2).sum()
    return J

data = pd.read_table('./ml_100k/u1.base',names=['user_id','item_id','rating','timestamp'])[['user_id','item_id','rating']]
train = defaultdict(lambda : defaultdict(lambda : 0))
train_r = defaultdict(lambda : defaultdict(lambda : 0))
print("generating training data")
for i in range(len(data)):
    row = data.iloc[i,:]
    train[row['item_id']][row['user_id']]=row['rating']
    

train_data = pd.DataFrame(train)
train_r_data = train_data.apply(lambda x:np.isnan(x)).apply(lambda x:x.astype(int)).fillna(0)
train_data = train_data.fillna(0)
train_r_m = np.matrix(train_r_data.as_matrix())
train_m = np.matrix(train_data.as_matrix())


