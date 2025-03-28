import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence,pad_sequence
from torch.autograd import Variable
import nni
import copy
import math
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy import interp
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar

def compared_version(ver1, ver2):
    """
    :param ver1
    :param ver2
    :return: ver1< = >ver2 False/True
    """
    list1 = str(ver1).split(".")
    list2 = str(ver2).split(".")
    
    for i in range(len(list1)) if len(list1) < len(list2) else range(len(list2)):
        if int(list1[i]) == int(list2[i]):
            pass
        elif int(list1[i]) < int(list2[i]):
            return -1
        else:
            return 1
    
    if len(list1) == len(list2):
        return True
    elif len(list1) < len(list2):
        return False
    else:
        return True


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if compared_version(torch.__version__, '1.5.0') else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TemporalEmbedding, self).__init__()

        self.absolute_position_embed=nn.Embedding(9999,d_model)
        self.year_embed = nn.Embedding(9999, d_model)
        self.month_embed = nn.Embedding(9999, d_model)

    def forward(self, x):
        x = x.long()

        month_x = self.month_embed(x[:, :, 1])
        year_x = self.year_embed(x[:, :, 0])
        abs_pos_x = self.absolute_position_embed(x[:, :, 2])
        
        return month_x + year_x + abs_pos_x

class IntervalEmbedding(nn.Module):
    def __init__(self, f_feature, d_model):
        super(IntervalEmbedding, self).__init__()

        self.interval_f = nn.Linear(1, f_feature)
        self.interval_r= nn.Linear(f_feature, d_model)
    
    def forward(self, x):
        
        f = 1-torch.tanh(self.interval_f(x)**2)
        x = self.interval_r(f)
        return x
    

class DataEmbedding(nn.Module):
    def __init__(self, f_feature, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        
        self.IntervalEmbedding = IntervalEmbedding(f_feature, d_model)
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        value_x=x[:,:,3:-1] 
        temporal_x=x[:,:,:3]
        interval_x=x[:,:,-1].view(x.shape[0],x.shape[1],-1)
        x = self.value_embedding(value_x) + self.temporal_embedding(temporal_x) + self.position_embedding(x) + self.IntervalEmbedding(interval_x)
        return self.dropout(x)
    
class mlp(torch.nn.Module):
    def __init__(self,num_i,num_h,num_o):
        super(mlp,self).__init__()
        
        self.linear1=torch.nn.Linear(num_i,num_h)
        self.relu=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(num_h,num_h) #2个隐层
        self.relu2=torch.nn.ReLU()
        self.linear3=torch.nn.Linear(num_h,num_o)
  
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x
    
class Transformer_encoder(nn.Module):
    def __init__(self,n_feature,f_feature,d_model,n_head,n_layer,n_linear,mlp_hidden,seq_len,dropout,class_num):
        super(Transformer_encoder,self).__init__()
        # self.input_embedding = Embeddings(embedding_dim, 999) 
        self.input_embedding = DataEmbedding(f_feature,n_feature, d_model)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=n_linear,dropout=dropout,batch_first=True,norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, n_layer)
        
        self.mlp = mlp(d_model*seq_len,mlp_hidden,mlp_hidden)
        self.decoder = nn.Linear(mlp_hidden,class_num)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def _generate_padding_mask(self,src):
        src_padding_mask = src.eq(0)[:,:,1].view(src.shape[0],-1)
        return src_padding_mask
    
    def forward(self,x):
        src=x
        src_padding_mask=self._generate_padding_mask(src)
        src = self.input_embedding(src)
        # src = self.pos_encoder(src)
        output = self.transformer_encoder(src,src_key_padding_mask=src_padding_mask).reshape(src.shape[0],-1)
        output = self.decoder(self.mlp(output))
        return output

def _generate_padding_mask(src):
    src_padding_mask = src.eq(0)[:,:,1].view(src.shape[0],-1)
    return src_padding_mask

def padding(data):
    data_x=torch.cat([F.pad(data[i],(0,0,0,11-data[i].shape[0])).unsqueeze(0) for i in range(len(data))])
    return data_x

def adjust_date_advanced(current_time=None, years=0, months=0, days=0, hours=0, minutes=0, seconds=0):
    """通过 datetime 和 calendar 实现更复杂的日期时间加减操作"""

    # 调整年份和月份
    new_year = current_time.year + years
    new_month = current_time.month + months

    while new_month > 12:
        new_year += 1
        new_month -= 12
    while new_month < 1:
        new_year -= 1
        new_month += 12

    # 处理日期溢出
    last_day = calendar.monthrange(new_year, new_month)[1]
    new_day = min(current_time.day, last_day)

    # 创建新的日期对象
    current_time = current_time.replace(year=new_year, month=new_month, day=new_day)

    # 调整剩余时间
    current_time += timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

    return current_time

def remove_nan(data):
    raw_data=data
    for i in range(raw_data.shape[0]):
        for j in range(raw_data.shape[1]):
            if np.isnan(raw_data[i,j]):
                raw_data[i,j]=0
    return raw_data

def from_csv_to_pregnancy_sequence(data):
    id_list=[i*7+5 for i in range(11)]
    test_data=[]
    for i in range(data.shape[0]):
        train=[]
        for j in range(len(id_list)):
            temp=[]
            if data[i,id_list[j]]==0:
                break
            temp.append(data[i,id_list[j]])  # 怀孕年
            temp.append(data[i,id_list[j]+1])  # 怀孕月
            temp.append(j+1) # abs_pos  
            temp.append(data[i,0]-data[i,1]) # current age
            temp.append(data[i,id_list[j]]-data[i,1]) # pregnance age
            temp.append(data[i,2]) # 民族
            temp.append(data[i,3]) # 受教育
            temp.append(data[i,4]) # 户口
            temp.append(data[i,id_list[j]+4]) # 存活
            temp.append(data[i,id_list[j]+2]) # 分娩
            temp.append(data[i,id_list[j]+3]) # 结局
            temp.append(round((12*(data[i,id_list[j]+5]-data[i,id_list[j]])+data[i,id_list[j]+6]-data[i,id_list[j]+1])/12,1)) # interval
            train.append(temp)
        train=torch.tensor(train,dtype=torch.float32)
        test_data.append(train)
    return test_data


super_params={
    'n_epochs':20,
    'n_feature':8,
    'class_num':4,
    'seq_len':11,
    # 'embedding_dim':2,
    'f_feature':32,
    'd_model':32,  # d_model = n_feature*embedding_dim
    'n_head':8,  # d_model % n_head == 0
    'n_layer':2,
    'n_linear':32,
    'mlp_hidden':16,
    'dropout':0.9,
    'lr':1e-2,
    'batch_size':2048,
    'gamma':2,
    'loss_type':'focal',
    'beta':0.2
}
device=torch.device('cuda:8' if torch.cuda.is_available() else 'cpu')
model=Transformer_encoder(super_params['n_feature'],super_params['f_feature'],super_params['d_model'],super_params['n_head'],super_params['n_layer'],super_params['n_linear'],super_params['mlp_hidden'],super_params['seq_len'],super_params['dropout'],super_params['class_num'])
for name, param in model.named_parameters():
    print(f"Parameter Name: {name}")
    print(f"Requires Grad: {param.requires_grad}")
    print(f"Parameter Value: {param}")
    print(f"Shape: {param.shape}")
    print("-" * 50)
model.load_state_dict(torch.load('parameter.pth'))
print('loading')
for name, param in model.named_parameters():
    print(f"Parameter Name: {name}")
    print(f"Requires Grad: {param.requires_grad}")
    print(f"Parameter Value: {param}")
    print(f"Shape: {param.shape}")
    print("-" * 50)
model=model.to(device)
model.eval()
# print(model)

if __name__ == '__main__':
    demo_data=pd.read_csv('demo.csv',header=None).values
    # demo_data_y=pd.read_csv('/data/0shared/liubo/population/final_figure_plot/demo/demo_label.csv',header=None).values
    demo_data=remove_nan(demo_data)

    test_data=from_csv_to_pregnancy_sequence(demo_data)
    test_data=padding(test_data).to(device)

    y_logits=model(test_data)
    soft_max=nn.Softmax(dim=1)
    y_score=soft_max(y_logits).cpu().detach().numpy()
    y_pred=np.argmax(y_score,axis=1)
    print(y_pred)
    print(f"The probability of four pregnancy outcome (live birth, stillbirth, spontaneous abortion, induced abortion) is {y_score}")
    outcome_list=["live birth", "stillbirth", "spontaneous abortion", "induced abortion"]
    print('The most likely outcome is ',outcome_list[y_pred])
