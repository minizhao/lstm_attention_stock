import pandas as pd
from torch.utils.data import DataLoader
from sklearn.cross_validation import train_test_split
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import  Variable
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from  sklearn import preprocessing  

batch_size=32
input_size=6
hidden_size=128
num_layers=1
cell_factory=nn.LSTM
x_days=20
y_days=1
learning_rate = 0.002

def gen_samples():
    data= pd.read_csv("hushen3006.csv")
    data=data[[x for x in data.columns if x!='date']]
    
    data_header=data.columns.values.tolist()
    x_data=[]
    y_data=[]
    ss=preprocessing.StandardScaler()
    data=ss.fit_transform(data)
   
    queue=[]
    for row in data:
        open_=row[data_header.index('open')]
        high_=row[data_header.index('high')]
        low_=row[data_header.index('low')]
        close_=row[data_header.index('close')]
        volume_=row[data_header.index('volume')]
        turn_=row[data_header.index('turn')]
        queue.append([open_,high_,low_,close_,volume_,turn_])
        if(len(queue)==(x_days+y_days)):
            x_data.append(queue[:-y_days])
            y_data.append([x[3] for x in queue[-y_days:]])# 3 is close position 
            queue=queue[1:]
            
    return x_data,y_data


class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim=None):
        """
        Attention mechanism
        :param enc_dim: Dimension of hidden states of the encoder h_j
        :param dec_dim: Dimension of the hidden states of the decoder s_{i-1}
        :param dec_dim: Dimension of the internal dimension (default: same as decoder).
        """
        super(Attention, self).__init__()

        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.attn_dim = self.dec_dim if attn_dim is None else attn_dim

        # W_h h_j
        self.encoder_in = nn.Linear(self.enc_dim, self.attn_dim, bias=False)
        self.decoder_in = nn.Linear(self.dec_dim, self.attn_dim, bias=False)
        self.att_linear = nn.Linear(self.attn_dim, 1, bias=False)

    def forward(self, dec_state, context, mask=None):
        """
        :param dec_state:  batch x dec_dim
        :param context: batch x T x enc_dim
        :return: Weighted context, batch x enc_dim
                 Alpha weights (viz), batch x T
        """
        batch, source_l, enc_dim = context.size()

        assert enc_dim == self.enc_dim

        # W*s over the entire batch (batch, attn_dim)
        dec_contrib = self.decoder_in(dec_state)
        
        # W*h over the entire length & batch (batch, source_l, attn_dim)
        enc_contribs = self.encoder_in(
            context.view(-1, self.enc_dim)).view(batch, source_l, self.attn_dim)

        # tanh( Wh*hj + Ws s_{i-1} )     (batch, source_l, dim)
        pre_attn = F.tanh(enc_contribs + dec_contrib.unsqueeze(1).expand_as(enc_contribs))

        # v^T*pre_attn for all batches/lengths (batch, source_l)
        energy = self.att_linear(pre_attn.view(-1, self.attn_dim)).view(batch, source_l)

        alpha = F.softmax(energy)
      
        weighted_context = torch.bmm(alpha.unsqueeze(1), context).squeeze(1)  # (batch, dim)

        return weighted_context, alpha


class att_rnn(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size,num_layers,cell_factory=nn.LSTM):
        super().__init__()
        
        self.encoder= cell_factory(input_size, hidden_size, num_layers, batch_first=True)
        self.Attention=Attention(enc_dim=hidden_size,dec_dim=hidden_size,attn_dim=64)
        self.decoder=cell_factory(1, hidden_size, num_layers, batch_first=True)
        self.hidden_to_out=nn.Linear(hidden_size, 1, bias=False)
        

    def forward(self,x,batch_size):
        
        hidden = (Variable(torch.zeros(num_layers,batch_size,hidden_size)).cuda(),
                  Variable(torch.zeros(num_layers,batch_size,hidden_size)).cuda()
                 )
        
        out_enc,(h_enc,c_enc)  = self.encoder(x, hidden)
        out_dec,(h_dec,c_dec)  = self.decoder(Variable(torch.zeros(batch_size,1).unsqueeze(2)).cuda(), hidden)
        
        weighted_context,_=self.Attention(c_dec.squeeze(0),out_enc.contiguous())
        
        y_=self.hidden_to_out(out_dec.squeeze(1))
        
        result_y=Variable(torch.zeros(batch_size,y_days)).cuda()
        for i in range(y_days):
            
            out_dec,(h_dec,c_dec)  = self.decoder(y_.unsqueeze(2), \
                                                  (h_dec,weighted_context.unsqueeze(0)))
            weighted_context,_=self.Attention(c_dec.squeeze(0),out_enc.contiguous())
            
            y_=self.hidden_to_out(out_dec.squeeze(1))
            result_y[:,i]=y_
            
        return result_y

    
def get_dev_loss(test_dataloader,model,criterion):
    loss_list=[]
    for batch in test_dataloader:
        x,y=Variable(batch[0]).float().cuda(),Variable(batch[1]).float()
        output=model(x,batch_size=batch_size)
        loss=criterion(output,y.cuda())/(batch_size*y_days)
        loss_list.append(loss.data[0])
    return np.mean(loss_list)

def save(model):
    save_filename = 'lstm_att.pt'
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)

def plot_show(x_data,y_data):
    model = torch.load('lstm_att.pt').cuda()
    plt.figure()
    if y_days>1:
        close_price=[x[0] for x in y_data]
    else:
        close_price=[x for x in y_data]
    plt.plot(list(range(len(close_price))), close_price, color='b')
    pred_price=[]
    for x_input in x_data:
        x_input=torch.from_numpy(np.array(x_input))
        output=model(Variable(x_input.unsqueeze(0)).float().cuda(),batch_size=1)
        pred_price.append(output.squeeze(0)[0].data[0])
    plt.plot(list(range(len(pred_price))), pred_price, color='r')
    plt.show()
        

def train(train_dataloader,test_dataloader,epochs=500):
    
    if os.path.isfile('lstm_att.pt'):
        model = torch.load('lstm_att.pt').cuda()
    else:
        model=att_rnn(batch_size,input_size,hidden_size,num_layers,cell_factory).cuda()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        loss_list=[]
        for batch in train_dataloader:
            x,y=Variable(batch[0]).float().cuda(),Variable(batch[1]).float()
            output=model(x,batch_size=batch_size)
            model.zero_grad()
            loss=criterion(output,y.cuda())/(batch_size*y_days)
            loss_list.append(loss.data[0])
            loss.backward()
            optimizer.step()
        if(epoch%10==0):
            print('train avg MSE loss : {}'.format(np.mean(loss_list)))
            print('test avg MSE loss : {}'.format(get_dev_loss(test_dataloader,model,criterion)))
            save(model)
            print('-------')
            
   
            
if __name__ == '__main__':
    x_data,y_data=gen_samples()
   
    train_X,test_X,train_Y,test_Y= train_test_split(x_data, y_data,test_size=0.25, random_state=33)
    train_dataloader=DataLoader(list(zip(np.array(train_X),np.array(train_Y))),batch_size=batch_size,\
                                shuffle=True,drop_last=True)
    test_dataloader=DataLoader(list(zip(np.array(test_X),np.array(test_Y))),batch_size=batch_size,\
                               shuffle=True,drop_last=True)

    
    #train(train_dataloader,test_dataloader)
    plot_show(x_data,y_data)
   
