# -*- coding: utf-8 -*-

import pandas as pd
from collections import defaultdict
import numpy as np

data = pd.read_table('./ml_100k/u1.base',names=['user_id','item_id','rating','timestamp'])[['user_id','item_id','rating']]
data2 = pd.read_table('./ml_100k/u1.test',names=['user_id','item_id','rating','timestamp'])[['user_id','item_id','rating']]
train_data = data.pivot_table(values='rating',index='user_id',columns='item_id')
train_r_data = (~train_data.isnull()).astype(int)
train_set = np.matrix(train_data.fillna(0).as_matrix())
train_r = np.matrix(train_r_data.as_matrix())
    
item_set = set(data['item_id'].values)
user_set = set(data['user_id'].values)

t = data2.apply(lambda x:(x['user_id'] in user_set) and (x['item_id'] in item_set),axis=1)
data2 = data2[t]


test_data = train_data.to_dict()
test_r = train_r_data.to_dict()
for item in test_data:
    for user in test_data[item]:
        test_data[item][user]=0
        test_r[item][user]=0
        
        
for i in range(len(data2)):
    row = data2.iloc[i,:]
    test_data[row['item_id']][row['user_id']] = row['rating']
    test_r[row['item_id']][row['user_id']] = 1

test_data = pd.DataFrame(test_data)
test_r = pd.DataFrame(test_r)
test_set = np.matrix(test_data.fillna(0).as_matrix())
test_r = np.matrix(test_r.as_matrix())
