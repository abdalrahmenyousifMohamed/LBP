import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    
    def __init__(self):
        super(Model,self).__init__()
        
        # number of input feature
        self.layer_1 = nn.Linear(12,512)
        self.layer_2 = nn.Linear(512,256)
        self.layer_out = nn.Linear(256 , 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.3)
        self.norm1 = nn.BatchNorm1d(512)
        self.norm2 = nn.BatchNorm1d(256)
    
    def forward(self , inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.norm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x