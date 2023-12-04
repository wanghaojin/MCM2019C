import model
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from torch import optim

def eval(y_pre,label):
    acc = 0
    for i in range(y_pre.shape[0]):
        for j in range(y_pre.shape[1]):
            acc += (y_pre[i][j] - label[i][j])**2
    acc = acc/(y_pre.shape[0]*y_pre.shape[1])
    return acc
data_by_year = {}
for year in range(2010, 2018):
    file_name = f'data_drug_{year}.npy'
    drug = np.load(file_name)
    data_by_year[year] = torch.tensor(drug).float()

adjacency = np.load('adjacent_data.npy')
tensor_adjacency = torch.tensor(adjacency).float()

alpha = 0.6
minloss = float('inf')
mlp = model.Mlp(alpha)
optimizer = optim.Adam(mlp.parameters(),lr = 0.02)
for epoch in range(400):
    for year in range(2010, 2016):
        optimizer.zero_grad()
        y_predicted = mlp(data_by_year[year],tensor_adjacency)
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_predicted, data_by_year[year+1])
        loss.backward()
        optimizer.step()
    if(epoch % 50 == 0):
        y_predicted = mlp(data_by_year[2016],tensor_adjacency)
        totalloss = eval(y_predicted,data_by_year[2017])
        print('totalloss:',totalloss,'\nminloss:',minloss)
        if(totalloss < minloss):
            minloss= totalloss
            y_2018 = mlp(data_by_year[2017],tensor_adjacency).detach().cpu().numpy()
            np.save('result.npy', y_2018)

