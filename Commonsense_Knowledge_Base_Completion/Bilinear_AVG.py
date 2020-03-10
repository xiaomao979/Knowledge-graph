# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:45:34 2020

@author: yuansiyu
"""
import torch 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
import os
import sys
import math
from collections import Counter
import numpy as np
import random
from torchtext.data import Field
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle

class preprocess():
    def __init__(self):
        self.train_triples = []
        self.valid_triples = []
        self.test_triples = []
        self.maxlen_s = 0
        self.maxlen_o = 0
        self.word_vec_map = {}
        self.word_id_map = {}
        self.embedding_weights = {}
        self.rel = {}
        self.n_r = 0

    def read_train_triples(self, filename):
        self.train_triples = [line.strip().split('\t') for line in open(filename,'r',encoding='UTF-8')]

    def read_valid_triples(self, filename):
        self.valid_triples = [line.strip().split('\t') for line in open(filename,'r',encoding='UTF-8')]

    def read_test_triples(self, filename):
        self.test_triples = [line.strip().split('\t') for line in open(filename,'r',encoding='UTF-8')]

    def read_relations(self, filename):
        self.rel ={line.strip():key+1 for key,line in enumerate(open(filename,'r',encoding='UTF-8'))}
        self.rel['UUUNKKK'] = 0
        self.n_rel = len(self.rel)

    def embedding_matrix(self):
        matrix = np.zeros((len(self.embedding_weights), self.embedding_dim))
        for key,value in self.embedding_weights.items():
            matrix[key,:] = value
        return matrix

    def load_embedding(self, filename):
        with open(filename,'r',encoding='UTF-8') as f:
            for i,line in enumerate(f):
                pair = line.split()
                word = pair[0]
                embedding = np.array(pair[1:], dtype='float32')
                self.word_vec_map[word] = embedding
                self.word_id_map[word] = i+1
            self.embedding_dim = len(embedding)

    def pretrained_embeddings(self, filename='embeddings.txt'):
        self.load_embedding(filename)
        for word, id_ in self.word_id_map.items():
            self.embedding_weights[id_] = self.word_vec_map[word]
        self.word_id_map['PAD'] = 0
        self.embedding_weights[self.word_id_map['PAD']] = np.zeros(self.embedding_dim)

    def sentence2idx(self, sentence):
        return [self.word_id_map[word] if word in self.word_id_map else self.word_id_map['UUUNKKK'] for word in sentence]

    def rel2idx(self, rel):
        return self.rel[rel.lower()] if rel.lower() in self.rel.keys() else self.rel['UUUNKKK'] 

    def	triple_to_index(self, triples, dev=False):
        triple2idx = []
        for triple in triples:
            if dev:
                p, s, o, label = triple[0], triple[1].split(' '), triple[2].split(' '), int(triple[3])
                triple2idx.append([self.rel2idx(p),self.sentence2idx(s),self.sentence2idx(o), label])
            else:
                p, s, o = triple[0], triple[1].split(' '), triple[2].split(' ')
                triple2idx.append([self.rel2idx(p),self.sentence2idx(s),self.sentence2idx(o)])
        return triple2idx

    def get_max_len(self, tripleidx):
        for triple in tripleidx:
            self.maxlen_s = max(self.maxlen_s, len(triple[1]))
            self.maxlen_o = max(self.maxlen_o, len(triple[2]))

    def pad_idx_data(self, tripleidx, dev=False):
        all_s, all_p, all_o = [], [], []
        if dev:
            label =[]
        for triple in tripleidx:
            pad_s = triple[1] + [self.word_id_map['PAD']]*(self.maxlen_s - len(triple[1]))
            all_s.append(pad_s)

            pad_o = triple[2] + [self.word_id_map['PAD']]*(self.maxlen_o - len(triple[2]))
            all_o.append(pad_o)

            all_p.append(triple[0])
            if dev:
                label.append(triple[3])
        if dev:
            return np.array(all_s), np.array(all_o), np.array(all_p), np.array(label)

        return np.array(all_s), np.array(all_o), np.array(all_p)


class  WordAVGModel(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, pad_idx, batch_size, pretrained_weights, n_r):
        super(WordAVGModel, self).__init__()
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.pretrained_weights = pretrained_weights
        self.embed = nn.Embedding(len(pretrained_weights), embedding_size, padding_idx = pad_idx)
        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.embed_rel = nn.Embedding(n_r, hidden_size*hidden_size, padding_idx = pad_idx)
        
    def forward(self, rel, sub, obj):
        self.embed.weight.data.copy_(torch.from_numpy(self.pretrained_weights))
        v1 = self.embed(sub).sum(dim=1) #[batch_size, embedding_size]
        v2 = self.embed(obj).sum(dim=1) #[batch_size, embedding_size]
        
        u1 = F.logsigmoid(self.fc1(v1))
        u1 = u1.unsqueeze(1)
        #print(u1.size())
        u2 = F.logsigmoid(self.fc1(v2))
        u2 = u2.unsqueeze(2)
        #print(u2.size())
        R = self.embed_rel(rel)
        #print(R.size())
        R = R.view(-1, self.hidden_size, self.hidden_size)
        #print(R.size())
        score = torch.bmm(u1, R)
        score = torch.bmm(score, u2)
    
        return F.sigmoid(score)
    

def binary_accuracy(preds,y):
    rounded_preds = torch.round(preds)#0,1prob
    correct = (rounded_preds == y).float()
    acc = correct.sum()/len(correct)
    return acc

def train(model, train_loader, optimizer, loss_fn):
    epoch_loss, epoch_acc = 0.,0.
    model.train()
    total_len = 0.
    for i, train_positive_data in enumerate(train_loader,0):
        train_s, train_o, train_p = train_positive_data
        train_negative_data = sample_negatives(train_positive_data)
        train_neg_s, train_neg_o, train_neg_p = train_negative_data
        train_label = np.concatenate((np.ones(len(train_s)), np.zeros(len(train_neg_s))))
        train_s = np.vstack([train_s, train_neg_s])
        train_o = np.vstack([train_o, train_neg_o])
        train_p = np.concatenate((train_p, train_neg_p))
        train_s, train_o, train_p, train_label = shuffle(train_s, train_o, train_p, train_label, random_state=4086)
        train_s = torch.from_numpy(train_s).long().to(device) 
        train_o = torch.from_numpy(train_o).long().to(device) 
        train_p = torch.from_numpy(train_p).long().to(device) 
        train_label = torch.from_numpy(train_label).to(device)
        preds = model(train_p, train_s, train_o).squeeze()
        train_label = train_label.type_as(preds)
        
        #print(preds)
        #print(train_label)
        loss = loss_fn(preds, train_label)
        
        acc = binary_accuracy(preds, train_label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * len(train_label)
        epoch_acc += acc.item() * len(train_label)
        total_len += len(train_label) 
        
    return epoch_loss / total_len, epoch_acc / total_len

def evaluate(model, train_loader, loss_fn):
    epoch_loss, epoch_acc = 0., 0.
    model.eval()
    total_len = 0.
    for i, train_positive_data in enumerate(train_loader,0):
        train_s, train_o, train_p = train_positive_data
        train_negative_data = sample_negatives(train_positive_data)
        train_neg_s, train_neg_o, train_neg_p = train_negative_data
        train_label = np.concatenate((np.ones(len(train_s)), np.zeros(len(train_neg_s))))
        train_s = np.vstack([train_s, train_neg_s])
        train_o = np.vstack([train_o, train_neg_o])
        train_p = np.concatenate((train_p, train_neg_p))
        train_s, train_o, train_p, train_label = shuffle(train_s, train_o, train_p, train_label, random_state=4086)
        train_s = torch.from_numpy(train_s).long().to(device) 
        train_o = torch.from_numpy(train_o).long().to(device) 
        train_p = torch.from_numpy(train_p).long().to(device) 
        train_label = torch.from_numpy(train_label).to(device)
        preds = model(train_p, train_s, train_o).squeeze()
        loss = loss_fn(preds, train_label)
        
        acc = binary_accuracy(preds, train_label)
        
        epoch_loss += loss.item() * len(train_label)
        epoch_acc += acc.item() * len(train_label)
        total_len += len(train_label) 
        
    model.train()
    return epoch_loss / total_len, epoch_acc / total_len

class TripleDataset():

    def __init__(self, data, dev=False):
        self.dev = dev
        if self.dev:
            self.s, self.o, self.p, self.label = data[0], data[1], data[2], data[3]
        else:
            self.s, self.o, self.p = data[0], data[1], data[2]
        self.len = len(self.s)

    def __getitem__(self, index):
        if self.dev:
            return self.s[index], self.o[index], self.p[index], self.label[index]	
        return self.s[index], self.o[index], self.p[index]

    def __len__(self):
        return self.len

def sample_negatives(data, type='RAND',sampling_factor=10):
    s_data, o_data, p_data = data[0], data[1], data[2]
    data_len = len(s_data)
    corrupt_s, corrupt_o = [], []
    true_s, true_o, true_p = [], [], []
    while(sampling_factor):
        for i in range(data_len):
            idx_s = random.randint(0,data_len-1)

            while i == idx_s: idx_s = random.randint(0,data_len-1)
            corrupt_s.append(s_data[idx_s].numpy())
            true_s.append(s_data[i].numpy())

            idx_o = random.randint(0,data_len-1)

            while i == idx_o: idx_o = random.randint(0,data_len-1)
            corrupt_o.append(o_data[idx_o].numpy())
            true_o.append(o_data[i].numpy())
            true_p.append(p_data[i].numpy())

        sampling_factor -= 1
    corrupt_s = np.array(corrupt_s)
    corrupt_o = np.array(corrupt_o)
    true_s = np.array(true_s)
    true_o = np.array(true_o)
    true_p = np.array(true_p)

    negative_s = np.vstack([corrupt_s, true_s])
    negative_o = np.vstack([true_o, corrupt_o])
    negative_p = np.concatenate((true_p, true_p))

    return torch.from_numpy(negative_s), torch.from_numpy(negative_o), torch.from_numpy(negative_p)

if __name__ == '__main__': 
    SEED = 1234
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_file = 'train100k.txt'
    valid_file = 'dev1.txt'
    rel_file = 'rel.txt'
    pretrained_file = 'embeddings.txt'

    preprocessor = preprocess()
    preprocessor.read_train_triples(train_file)
    preprocessor.read_relations(rel_file)
    preprocessor.pretrained_embeddings(filename=pretrained_file)
    n_r = preprocessor.n_rel
    train_triples = preprocessor.train_triples
    train_idx = preprocessor.triple_to_index(train_triples)
    preprocessor.get_max_len(train_idx)
    train_data = preprocessor.pad_idx_data(train_idx)

    pretrained_weights = preprocessor.embedding_matrix()
    embedding_size = preprocessor.embedding_dim
    word_id_map = preprocessor.word_id_map
    rel_id_map = preprocessor.rel

    # Prepare Validation DataSet
    preprocessor.read_valid_triples(valid_file)
    valid_triples = preprocessor.valid_triples
    valid_idx = preprocessor.triple_to_index(valid_triples, dev=True)
    valid_data = preprocessor.pad_idx_data(valid_idx, dev=True)
    valid_label = valid_data[3]
    valid_data = valid_data[0], valid_data[1], valid_data[2]

    hidden_size = 20
    output_size = 1
    batch_size = 32
    pad_idx = 1
    train_loader = DataLoader(TripleDataset(train_data), batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(TripleDataset(valid_data), batch_size=batch_size, shuffle=True, num_workers=0)
    model = WordAVGModel(embedding_size, hidden_size, output_size, pad_idx, batch_size, pretrained_weights, n_r)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.BCEWithLogitsLoss()#二分类且无需转换的熵损失
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    num_epochs = 5
    best_valid_acc = 0.
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, loss_fn)
        print("epoch",epoch,"train:",train_loss, train_acc)
        valid_loss, valid_acc = evaluate(model, valid_loader, loss_fn)
        print("epoch",epoch,"valid:",valid_loss, valid_acc)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), 'bilinear_wordavg_model.pth')


    test_file = 'test.txt'
    preprocessor.read_test_triples(test_file)
    test_triples = preprocessor.test_triples
    test_idx = preprocessor.triple_to_index(test_triples, dev=True)
    test_data = preprocessor.pad_idx_data(test_idx, dev=True)
    test_label = test_data[3]
    test_data = test_data[0], test_data[1], test_data[2]
    test_loader = DataLoader(TripleDataset(test_data), batch_size=batch_size, shuffle=True, num_workers=0)
    test_loss, test_acc = evaluate(model, test_loader, loss_fn)
    print("test:",test_loss, test_acc)
     




