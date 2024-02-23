import torch
import torch.nn as nn
import torch.nn.functional as F

class S2VT_model(torch.nn.Module):
    def __init__(self, vocab_size, batch_size, frame_dim, hidden_dim, dropout, f_len, device, caption_length):
        super(S2VT_model, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.frame_dim = frame_dim
        self.hidden_dim = hidden_dim
        self.f_len = f_len
        self.caption_length = caption_length

        self.drop = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(frame_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(2*hidden_dim, hidden_dim, batch_first=True)

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
    def forward(self, feat, cap):
        feat = self.linear1(self.drop(feat.contiguous().view(-1, self.frame_dim)))
        feat = feat.view(-1, self.f_len, self.hidden_dim)
        padding = torch.zeros([feat.shape[0], self.caption_length-1, self.hidden_dim]).to(self.device)
        feat = torch.cat((feat, padding), 1)
        
        l1, h1 = self.lstm1(feat)
        
        cap = self.embedding(cap[:,:self.caption_length-1])
        padding = torch.zeros([feat.shape[0], 80, self.hidden_dim]).to(self.device)
        cap = torch.cat((padding, cap), 1)
        cap = torch.cat((cap, l1), 2) # bs, 120, 1024

        l2, h2 = self.lstm2(cap) # 32, 120, 512
        
        l2 = l2[:, self.f_len:, :] # batch_size, 40, hidden
        l2 = self.drop(l2.contiguous().view(-1, self.hidden_dim)) # 1280, 512 (contig)
        output = F.log_softmax(self.linear2(l2), dim=1) # 1280, 2046
        return output
    
    def test(self, feat):
        caption = []
        feat = self.linear1(self.drop(feat.contiguous().view(-1, self.frame_dim)))
        feat = feat.view(-1, self.f_len, self.hidden_dim)
        padding = torch.zeros([feat.shape[0], self.caption_length-1, self.hidden_dim]).to(self.device)
        feat = torch.cat((feat, padding), 1)
        l1, h1 = self.lstm1(feat)
        
        padding = torch.zeros([feat.shape[0], self.caption_length-1, self.hidden_dim]).to(self.device)
        cap_in = torch.cat((padding, l1), 1)
        l2, h2 = self.lstm2(cap_in)
        
        bos = torch.ones(self.batch_size).to(self.device)
        cap_in = self.embedding(bos)
        cap_in = torch.cat((cap_in, l1[:,80,:]),1).view(self.batch_size, 1, 2*self.hidden_dim)
        
        l2, h2 = self.lstm2(cap_in, h2)
        l2 = torch.argmax(self.linear2(self.drop(l2.contiguous().view(-1, self.hidden_dim))),1)
        
        caption.append(l2)
        for i in range(self.f_len-2):
            cap_in = self.embedding(l2)
            cap_in = torch.cat((cap_in, l1[:, self.f_len+1+i, :]), 1)
            cap_in = cap_in.view(self.batch_size, 1, 2* self.hidden_dim)
            l2, h2 = self.lstm2(cap_in, h2)
            l2 = l2.contiguous().view(-1, self.hidden_dim)
            l2 = torch.argmax(self.linear2(self.drop(l2)),1)
            caption.append(l2)
        return caption