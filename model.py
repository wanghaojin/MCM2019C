import torch
import torch.nn as nn
import numpy

class Mlp(nn.Module):
    def __init__(self,alpha):
        super().__init__()
        self.z1 = nn.Linear(461,69)
        self.layer1= nn.Linear(69,16)
        self.relu = nn.Sigmoid()
        self.layer2 = nn.Linear(16,69)
        self.alpha = alpha
    def forward(self,X,A):
        X = X.float()
        A = A.float()
        A_filtered = self.z1(A)
        Z = self.alpha*X + (1-self.alpha)*A_filtered
        tem = self.layer1(Z)
        ans = self.relu(tem)
        return self.layer2(ans)

