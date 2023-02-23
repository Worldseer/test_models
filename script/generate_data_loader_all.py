import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
AADICT = {'A':1, 'R':2, 'N':3, 'D':4, 'C':5, 'Q':6, 'E':7, 'G':8, 'H':9, 'I':10,
    'L':11, 'K':12, 'M':13, 'F':14, 'P':15, 'S':16, 'T':17, 'W':18, 'Y':19, 'V':20}
    

def effectivelen(seq,W):
    for i in range(3,W+1,1):
        if i*i < len(seq) and i!=W:
            continue
        else:
            break
    return i

def sequence2num(sequence):
    data = []
    for acid in sequence:
        data.append(int(AADICT.get(acid,'0')))
    return torch.tensor(data) 

def expand_sequence(sequence,expand_len):
    if len(sequence) > expand_len:
        return sequence[:expand_len]
    else:
        return sequence+"*"*(expand_len-len(sequence))
        
def genimatrix1(sequence,out_size): #x,y are the coordinates of the current data, the default value is (0,0), and the sequence is the sequence that has been converted to a numerical representation.
    W,H = effectivelen(sequence,out_size),effectivelen(sequence,out_size)
    matrix = torch.zeros((out_size,out_size))
    if len(sequence) >W*H: 
        sequence = sequence[0:W*H]
    seqs = sequence2num(sequence)
    x,y = 0,0
    step = 0
    for seq in seqs:
        if step<W:
            matrix[x,y] = seq
            y += 1
            step += 1             
        if step == W:
            step = 0
            y = 0
            x += 1
            continue  
    return torch.LongTensor(matrix.numpy())  


        
def genimatrix2(sequence,out_size): #x,y are the coordinates of the current data, the default value is (0,0), and the sequence is the sequence that has been converted to a numerical representation.
    flag = 0 
    W,H = effectivelen(sequence,out_size),effectivelen(sequence,out_size)
    matrix = torch.zeros((out_size,out_size))
    if len(sequence) >W*H: 
        sequence = sequence[0:W*H]
    seqs = sequence2num(sequence)
    x,y = 0,0
    step = 0
    for seq in seqs:
        if flag == 0:#right
            if step<W:
                matrix[x,y] = seq
                y += 1
                step += 1             
            if step == W:
                flag = 2
                step = 0
                y -= 1
                x += 1
                continue
        if flag == 2:#left
            if step<W:
                matrix[x,y] = seq
                y -= 1
                step += 1             
            if step == W:
                flag = 0
                step = 0
                y += 1
                x += 1
                continue      
    return torch.LongTensor(matrix.numpy())

def genimatrix3(sequence,out_size): 
        flag = 0
        W,H = effectivelen(sequence,out_size),effectivelen(sequence,out_size)
        matrix = torch.zeros((out_size,out_size))
        if len(sequence) >W*H: #Truncate if len>W*H
            sequence = sequence[0:W*H]
        x,y = 0,0
        step = 0
        seqs = sequence2num(sequence)
        for seq in seqs:
            if flag == 0:#right
                if step<W:
                    matrix[x,y] = seq
                    y += 1
                    step += 1             
                if step == W:
                    flag = 1 
                    step = 0
                    H -= 1
                    y -= 1
                    x += 1
                    continue
            if flag == 1:#down
                if step<H:
                    matrix[x,y] = seq
                    x += 1
                    step += 1
                if step == H:
                    flag = 2
                    step = 0
                    W -= 1
                    x -= 1
                    y -= 1 
                    continue
            if flag == 2:#left
                if step<W:
                    matrix[x,y] = seq
                    y -= 1
                    step += 1
                if step == W:
                    flag = 3 
                    step = 0
                    H -= 1
                    y += 1
                    x -= 1
                    continue
            if flag == 3:#up
                if step<H:
                    matrix[x,y] = seq
                    x -= 1
                    step += 1
                if step == H:
                    flag = 0
                    step = 0
                    W -= 1
                    x += 1
                    y += 1          
        return torch.LongTensor(matrix.numpy())



def list_to_dict(go_list):
    go_dict = dict()
    for i,goid in enumerate(go_list):
        go_dict[goid] = i
    return go_dict     

    
def token2num(seq):#output torch.out_size([2304])
    data = torch.zeros(len(seq),dtype=torch.int32)
    for idx,acid in enumerate(seq):
        data[idx] = int(AADICT.get(acid,'0'))
    return data    

    
class traindataset(Dataset):
    def __init__(self,protein_df,go_df,out_size):
        super(traindataset,self).__init__()
        self.protein_df = protein_df
        self.go_dict = list_to_dict(go_df.terms.values)
        self.length = len(protein_df)
        self.out_size = out_size

        
    def __len__(self):
        return self.length
    
    def __getitem__(self,idx):
        labels = torch.zeros(len(self.go_dict), dtype=torch.float32)
        seq = self.protein_df.iloc[idx].sequences# Extraction of the corresponding protein sequence
        expand_seq = expand_sequence(seq,self.out_size*self.out_size)
        # emb_data = token2num(expand_seq)
        imatrix1  = genimatrix1(seq,self.out_size)
        imatrix2  = genimatrix2(seq,self.out_size)
        imatrix3  = genimatrix3(seq,self.out_size)
        imatrix4  = genimatrix1(expand_seq,self.out_size)
        imatrix5  = genimatrix2(expand_seq,self.out_size)
        imatrix6  = genimatrix3(expand_seq,self.out_size)       
        for go in self.protein_df.iloc[idx].annotations:
            index = self.go_dict.get(go)
            if index != None:
                labels[index] = 1 
        return [imatrix1,imatrix2,imatrix3,imatrix4,imatrix5,imatrix6],labels
        


def trainloader(data_df,go_df,imageout_size,batchout_size,shuffle=True):# input df data, output DataLoader
    dataloader = DataLoader(dataset=traindataset(data_df,go_df,imageout_size),shuffle=shuffle,batch_size=batchout_size,num_workers=6)
    return dataloader

  
