from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import gensim
import spacy 

nlp = spacy.load('en_core_web_sm') 

from nltk import word_tokenize
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

import os
import re
import random
import nltk
from gensim.models import Word2Vec

def word2vec_(s,p,s1,p1,hidden_size):
	with open(s) as f,open(p) as g,open(s1) as f1,open(p1) as g1:
		st=[]
		gh = f.readlines()
		gt = g.readlines()
		for line in gh:
			list1= (line.strip()).split()
			st.append(list1)
		for line in gt:
			list1= (line.strip()).split()
			st.append(list1)
		gh1 = f1.readlines()
		gt1 = g1.readlines()
		for line in gh1:
			list1= (line.strip()).split()
			st.append(list1)
		for line in gt1:
			list1= (line.strip()).split()
			st.append(list1)
	fname = 'word2vec.model'
	model = Word2Vec(st,size=hidden_size,window=5, min_count=1)
	#model.train(st,total_examples=len(st),epoch = 10)
	#model.save(fname)
	return model

model = word2vec_('para100','ques100','para','ques',100)


pos_feat_size = 46

pos_list = ['LS', 'TO', 'VBN', "''", 'WP', 'UH', 'VBG', 'JJ', 'VBZ', '--', 'VBP', 'NN', 'DT', 'PRP', ':', 'WP$', 'NNPS', 'PRP$', 'WDT', '(', ')', '.', ',', '``', '$', 'RB', 'RBR', 'RBS', 'VBD', 'IN', 'FW', 'RP', 'JJR', 'JJS', 'PDT', 'MD', 'VB', 'WRB', 'NNP', 'EX', 'NNS', 'SYM', 'CC', 'CD', 'POS']
def get_postag_(words):
    vect = np.zeros((len(words),len(pos_list)+1))
    try:
        temp = list(zip(*(nltk.pos_tag(words))))[1]
        for rd in range(len(temp)):
            vect[rd][pos_list.index(temp[rd])] = 1
    except:
        print('postag exception [look for #]')
        print('\n')
    return vect

ner_feat_size = 19
ner_tags = ['ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'EVENT', 'WORK_OF_ART', 'MONEY', 'FAC', 'QUANTITY', 'GPE', 'DATE', 'CARDINAL', 'LANGUAGE', 'PRODUCT', 'LAW', 'NORP', 'TIME', 'LOC']
def get_nertag(words):
    tagr = []
    st = " ".join(words) 
    txt = []
    lab = []
    for ent in nlp(st).ents:
        sp = ent.text.split()
        for i in range(len(sp)):
            txt.append(sp[i])
            lab.append(ent.label_)
    ch = 0
    for i in words:
        vect = np.zeros(len(ner_tags)+1)
        if i in txt:
            vect[ner_tags.index(lab[ch])] = 1
        tagr.append(vect)
    return tagr
    

def extra_indices(count,voc,para,words):
    vocu = voc.copy()
    cont = []
    extra_zeros = 0
    for i in para:
        if i not in list(vocu.keys()):
            vocu[i] = count+extra_zeros
            cont.append(count+extra_zeros)
            extra_zeros+=1
        else:
            cont.append(vocu[i])
    trg = [] 
    for i in words:
        try:
            trg.append(vocu[i])
        except:
            print("out of vocab in queries",i)
    return cont,trg,extra_zeros,vocu

hidden_dim = 50
hidden_size = 100 + pos_feat_size + ner_feat_size
hidden_sizeq = 100
batch_size = 1
layer_dim = 1

def embedding_one_hot(unique_vocab):
    word = {}
    for i in range(len(unique_vocab)):
        t = np.zeros((len(unique_vocab)))
        t[i] =1
        word[unique_vocab[i]] = t
    return word

def indices(count,voc,words,emb):
    prsnt_emb = []
    for i in words:
        if(i not in voc.keys()):
            voc[i] = count
            count+=1
            try:
                emb[i] = model[i]
            except:
                print(i,"didnt found this word emb")
        prsnt_emb.append(emb[i])
#             prsnt_emb.append(emb[i])
            
    return prsnt_emb,count,voc,emb

def indices_(count,voc,words,emb):
    prsnt_emb = []
    for i in words:
        if(i not in voc.keys()):
            voc[i] = count
            count+=1
            try:
                emb[i] = model[i]
            except:
                print(i,"didnt found this word emb")
        prsnt_emb.append(emb[i])
    
    ner_feat = get_nertag(words)
    pos_feat = get_postag_(words)
    re = torch.cat((torch.tensor(prsnt_emb),torch.tensor(pos_feat).float()),1)
    rer = torch.cat((re,torch.tensor(ner_feat).float()),1)
    
#             prsnt_emb.append(emb[i])
    return rer,count,voc,emb        

def get_data(file_path1,file_path2,file_path3,file_path4,file_path5,file_path6,vocp,vocq,countp,countq,embp,embq):
    with open(file_path1) as f1, open(file_path2) as f2, open(file_path3) as f3, open(file_path4) as f4, open(file_path5) as f5, open(file_path6) as f6:
        para_sent = f1.readlines()
        ques_sent = f2.readlines()
        spant = f3.readlines()
        para_sen = f4.readlines()
        ques_sen = f5.readlines()
        span = f6.readlines()
        
        
    para_embt= []
    ques_indt= []
    cont_wt= []
    extra_0t= []
    ques_embt= []
    startt= []
    endt= []
    para_emb = []
    ques_ind = []
    cont_w = []
    extra_0 = []
    ques_emb = []
    start = []
    ext_voc = []
    end = []
    
    vecte = np.zeros(len(pos_list)+1)
    
    for i in  range(len(para_sent)):
        prsnt_embp,countp,vocp,embp = indices_(countp,vocp,(para_sent[i].strip()).split(),embp)
        prsnt_embq,countq,vocq,embq = indices(countq,vocq,(ques_sent[i].strip()).split(),embq)
        prsnt_embq.insert(0,embq['SOS'])
        prsnt_embq.insert(len(prsnt_embq),embq['EOS'])
        para_embt.append(prsnt_embp)
        ques_embt.append(prsnt_embq)
        temp = (spant[i].strip()).split(' ')
        startt.append(int(temp[0]))
        endt.append(int(temp[1]))
    print("finn")
    
    print(len(para_embt[0]),len(para_embt[0][0]))
    print(len(ques_embt[0]),len(ques_embt[0][0]))
        
    for i in  range(len(para_sen)):
        prsnt_embp,countp,vocp,embp = indices_(countp,vocp,(para_sen[i].strip()).split(),embp)
        prsnt_embq,countq,vocq,embq = indices(countq,vocq,(ques_sen[i].strip()).split(),embq)
        prsnt_embq.insert(0,embq['SOS'])
        prsnt_embq.insert(len(prsnt_embq),embq['EOS'])
        para_emb.append(prsnt_embp)
        ques_emb.append(prsnt_embq)
        temp = (span[i].strip()).split(' ')
        start.append(int(temp[0]))
        end.append(int(temp[1]))
        
    print("mid-stat")
    for i in range(len(para_sent)):
        cont,prsnt_vocq,ext,vocu = extra_indices(countq,vocq,(para_sent[i].strip()).split(),(ques_sent[i].strip()).split())
        cont_wt.append(cont)
        prsnt_vocq.insert(0,0)
        prsnt_vocq.insert(len(prsnt_vocq),1)
        extra_0t.append(ext)
        ques_indt.append(prsnt_vocq)
        
    for i in range(len(para_sen)):
        cont,prsnt_vocq,ext,vocu = extra_indices(countq,vocq,(para_sen[i].strip()).split(),(ques_sen[i].strip()).split())
        cont_w.append(cont)
        prsnt_vocq.insert(0,0)
        prsnt_vocq.insert(len(prsnt_vocq),1)
        extra_0.append(ext)
        ques_ind.append(prsnt_vocq)
        ext_voc.append(vocu)
              
    return para_embt,ques_embt,ques_indt,startt,endt,extra_0t,cont_wt,para_emb,ques_emb,ques_ind,start,end,ext_voc,extra_0,cont_w,countp,countq

vocp = {"SOS":0,"EOS":1}
vocq = {"SOS":0,"EOS":1}
sosemb = torch.zeros(hidden_sizeq)
eosemb = torch.ones(hidden_sizeq)
embp = {"SOS":sosemb,"EOS":eosemb}
embq = {"SOS":sosemb,"EOS":eosemb}
countp = 2
countq = 2
para_embt,ques_embt,ques_indt,startt,endt,extra_0t,contt,para_emb,ques_emb,ques_ind,start,end,ext_voc,extra_0,cont,countp,countq = get_data("/home/mani/para","/home/mani/ques","/home/mani/ansr","/home/mani/para100","/home/mani/ques100","/home/mani/ansr100",vocp,vocq,countp,countq,embp,embq)


class contextR(nn.Module):
    def __init__(self, hidden_size , hidden_dim, layer_dim, batch_size):
        super(contextR, self).__init__()
    
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_size
        self.layer_dim = layer_dim
        self.batch_size = batch_size
#         self.embeddp = nn.Embedding(countp, hidden_size)
        self.lstm = nn.LSTM(hidden_size+1, hidden_dim, layer_dim, batch_first=True, bidirectional=True)


    def forward(self, x):
        
        inp_len = len(x)
#         x = self.embeddp(x)
        x = x.view(1,inp_len,self.hidden_size+1)  #+1 for adding extra features , here for answer
        h0 = torch.zeros(self.layer_dim*2, self.batch_size, self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim*2, self.batch_size, self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        return out , (hn,cn)
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_dim, device=device)

class queG(nn.Module):
    def __init__(self, hidden_size , hidden_dim, layer_dim, batch_size,countq):
        super(queG, self).__init__()
    
        self.lstm = nn.LSTM(hidden_size, hidden_dim, layer_dim, batch_first=True)
        
        self.hidden_size = hidden_size
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.batch_size = batch_size
    
        self.sigmoid = nn.Sigmoid()
        self.softlog = nn.LogSoftmax(dim=1)
        self.soft = nn.Softmax(dim=None)
#         self.w_h = torch.nn.Parameter(torch.randn(100, 50))
        self.w_h = nn.Linear(50,100)
        self.linear1 = nn.Linear(150,200)
        self.linear2 = nn.Linear(200,countq)
        self.linear3 = nn.Linear(50,1)
        self.linear4 = nn.Linear(50,1)

    def forward(self,H,y,h0,c0,cont,ext):
        
        y = y.view(1,1,self.hidden_size)
        y = y.float()
        out, (h0, c0) = self.lstm(y, (h0.detach(), c0.detach()))
        a_t = self.soft(H[0]@self.w_h(h0[0][0]))
        c_t = H[0].t()@a_t
        temp = torch.cat((h0[0][0],c_t),0)
        e_t = self.sigmoid(self.linear1(temp))
        out = self.linear2(e_t).view(1,-1)
        out = torch.cat((out,torch.zeros(1,ext)),1)
        p1 = self.sigmoid(self.linear3(h0[0][0].t()))
        p2 = self.sigmoid(self.linear4(h0[0][0].t()))
        out = p1*out
        attn = p2*a_t
        
        out = out.squeeze(0)
        out = out.scatter_add(0,torch.tensor(cont),attn)
        out = self.softlog(out.view(1,-1))
        return out,h0,c0
    
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_dim)

class seq(nn.Module):
    def __init__(self, encoder,decoder,countp,countq):
        super(seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
        self.lstm = nn.LSTM(65, hidden_dim, layer_dim, batch_first=True)
        
        self.countp = countp
        self.countq = countq

#         self.embeddp = nn.Embedding(countp, hidden_size)
#         self.embeddq = nn.Embedding(countq, hidden_size)
        
    def forward(self,inp,trg,mode,words,start,end,cont,ext):
        
        x = torch.tensor(inp).view(-1,hidden_size)
        
        x = x.float()
        
        x = torch.cat((x,(torch.zeros(len(x)).scatter_add(0,torch.tensor(range(start,end+1)),torch.ones(1+end-start))).view(-1,1)),dim=1)
        
        lis = []
        H, (h0,c0) = self.encoder(x)

        h0 = self.decoder.initHidden().requires_grad_()
        c0 = self.decoder.initHidden().requires_grad_()
        
        mod_x = x[:,100:-1][start:end+1]
        mod_x = mod_x.view(1,-1,65)
        
        _, (h0, c0) = self.lstm(mod_x, (h0.detach(), c0.detach()))
        
        if(mode == "train"):
            for j in range(len(trg)-1):
                out,h0,c0 = self.decoder(H,torch.tensor(trg[j]).view(1,hidden_sizeq),h0,c0,cont,ext)
#                 print(torch.argmax(out[0]),out)
                lis.append(out)

#         gold = []
            
        if(mode == "test"):
            
#             for ds in range(len(y)):
#                 gold.append(words[y[i]])
            
            for di in range(len(trg)-1):
                out,h0,c0 = self.decoder(H,torch.tensor(trg[di]).view(1,hidden_sizeq),h0,c0,cont,ext)
                index = torch.argmax(out[0])
#                 print(index)
                try:
                    lis.append(words[index])
#                 out1 = torch.tensor([[index.item()]])
                except:
                    print(index)
                if index == 1:
                    return lis

        return lis
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


enc = contextR(hidden_size,hidden_dim,layer_dim,batch_size)
dec = queG(hidden_sizeq,hidden_dim,layer_dim,batch_size,countq)
criterion = nn.NLLLoss()
model = seq(enc,dec,countp,countq)
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model,src_data,trg_data,trg_ind,optimizer,criterion,start,end,cont,extra_zeros):
    total_loss = 0
    for i in range(len(src_data)):
        optimizer.zero_grad()
        dup = []
        lis = model(src_data[i],trg_data[i],"train",dup,start[i],end[i],cont[i],extra_zeros[i])
        loss = 0
        for j in range(1,len(lis)+1):
#             print(torch.tensor(trg_data[i][j]),torch.tensor(lis[j-1]).shape)
            loss+= criterion(lis[j-1],torch.tensor([trg_ind[i][j]]))
        total_loss+=loss
        loss.backward()
        optimizer.step()
#     print(total_loss)

def test(model, input_sent,target_sent,target_ind,start,end,cont,extra_zeros,ext_voc):
    fh = open('quegcpu','a')
#     rscore = 0
    
    with torch.no_grad():

        for i in range(len(input_sent)):
            words = list(ext_voc[i].keys())
            
            gen = model(input_sent[i],target_sent[i],"test",words,start[i],end[i],cont[i],extra_zeros[i])
            
#             print(target_sent[i])
#             print(gen)
#             print('\n')
            fh.write(str(gen))
            fh.write('\n')
            
            
#             rscore += rouge_score([gold], gen)
            
#     print('ROUGE score',rscore/len(input_sent))



for i in range(30):
    train(model,para_embt,ques_embt,ques_indt,optimizer,criterion,startt,endt,contt,extra_0t)
    test(model, para_emb,ques_emb,ques_ind, start,end,cont,extra_0,ext_voc)

torch.save(model,'modelqgcpu')

