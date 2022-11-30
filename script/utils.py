from collections import deque, Counter
import pandas as pd
import numpy as np
from xml.etree import ElementTree as ET
import torch
import math

BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'
FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS}

NAMESPACES = {
    'cc': 'cellular_component',
    'mf': 'molecular_function',
    'bp': 'biological_process'
}

EXP_CODES = set([
    'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC'])

# CAFA4 Targets
CAFA_TARGETS = set([
    '287', '3702', '4577', '6239', '7227', '7955', '9606', '9823', '10090',
    '10116', '44689', '83333', '99287', '226900', '243273', '284812', '559292'])

def is_cafa_target(org):
    return org in CAFA_TARGETS

def is_exp_code(code):
    return code in EXP_CODES


class Ontology(object):

    def __init__(self, filename='data/go.obo', with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ic = None

    def has_term(self, term_id):
        return term_id in self.ont

    def get_term(self, term_id):
        if self.has_term(term_id):
            return self.ont[term_id]
        return None

    def calculate_ic(self, annots):
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])
            self.ic[go_id] = math.log(min_n / n, 2)
    
    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def load(self, filename, with_rels):
        ont = dict()
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['regulates'] = list()
                    obj['alt_ids'] = list()
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        # add all types of relationships
                        obj['is_a'].append(it[1])
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
            if obj is not None:
                ont[obj['id']] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)
        return ont


    def get_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set


    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set


    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']
    
    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set




    
#统计GO术语的数量，将小于某个阈值的GO术语排除，返回GO术语
#提取每个域中的数据
def get_domain_df(go_obo, df):
    go = Ontology(go_obo)
    mf = set()
    mfo = list()    
    bp = set()
    bpo = list()
    cc = set()
    cco = list()
    for i,items in df.iterrows():
        # if items.cafa_target == True:
        for goid in items.annotations:
            if goid in go.ont:
                if go.ont[goid]['namespace'] == 'molecular_function' and i not in mf:
                    mfo.append(items)
                    mf.add(i)
                if go.ont[goid]['namespace'] == 'biological_process' and i not in bp:
                    bpo.append(items)
                    bp.add(i)
                if go.ont[goid]['namespace'] == 'cellular_component' and i not in cc:
                    cco.append(items)
                    cc.add(i)
    mf_df = pd.DataFrame(mfo).reset_index(drop=True)
    bp_df = pd.DataFrame(bpo).reset_index(drop=True)
    cc_df = pd.DataFrame(cco).reset_index(drop=True)
    #df.to_pickle(out_pickle)
    return mf_df,bp_df,cc_df         
    
def split_train(df,valid_percent):
    # df = pd.read_pickle(train_pickle)
    n = len(df)#n表示有多少个数据
    index = np.arange(n)#获取每个数据的索引，以便进行shuffle
    valid_n = int(n * valid_percent)#计算训练数据的数量
    np.random.seed(seed=0)
    np.random.shuffle(index)
    valid_df = df.iloc[index[:valid_n]]
    train_df = df.iloc[index[valid_n:]]
    return train_df, valid_df    
 
#统计某个namespace中go术语存在的蛋白质数据数量
def compute_num(go_obo,namespce,df):
    go = Ontology(go_obo)
    cnt = Counter()
    for _,item in df.iterrows():
        for goterm in item.annotations:
            if goterm in go.ont:
                if go.get_namespace(goterm) == namespce:
                    cnt[goterm] += 1
    df = pd.DataFrame.from_dict(cnt,orient='index').reset_index()
    df.columns = ['terms','num']
    df = df.sort_values(by='num',ascending=False,ignore_index=True)
    return df    
 
 
#输入原始无划分domain的df,和go文件，实现划分数据集，输出goterm（少于某个阈值的排除），输出train_mf,valid_mf,train_bp,valid_bp,train_cc,valid_cc,go_mf,go_bp,go_cc
def get_domain(go_obo, df, limit_go_num = False, split=False, valid_percent=0.1,min_count=50):
    cnt = Counter()
    for i, row in df.iterrows():
        for term in row['annotations']:
            cnt[term] += 1   
    mf_df,bp_df,cc_df = get_domain_df(go_obo, df)
    go_mf = compute_num(go_obo,"molecular_function",mf_df)
    go_bp = compute_num(go_obo,"biological_process",bp_df)
    go_cc = compute_num(go_obo,"cellular_component",cc_df)
    if limit_go_num == True:
        go_mf = go_mf[go_mf.num>=min_count].reset_index(drop=True)
        go_bp = go_bp[go_bp.num>=min_count].reset_index(drop=True)
        go_cc = go_cc[go_cc.num>=min_count].reset_index(drop=True)
    if split == True:
        train_mf, valid_mf = split_train(mf_df,valid_percent)
        train_bp, valid_bp = split_train(bp_df,valid_percent)
        train_cc, valid_cc = split_train(cc_df,valid_percent)
        return train_mf, valid_mf, go_mf, train_bp, valid_bp, go_bp, train_cc, valid_cc, go_cc
        
    else:
    
        return mf_df, go_mf, bp_df, go_bp, cc_df, go_cc     
        
#从df中得到.fasta文件 
def get_fasta(df,targetpath):
    with open(targetpath,'w') as f:
        for idx,items in df.iterrows():
            f.writelines([">", items.proteins,"\n"])
            f.writelines([items.sequences,"\n"])

def read_fasta(filename):
    seqs = list()
    info = list()
    seq = ''
    inf = ''
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq != '':
                    seqs.append(seq)
                    info.append(inf)
                    seq = ''
                inf = line[1:]
            else:
                seq += line
        seqs.append(seq)
        info.append(inf)
    return info, seqs

def predict_df(data_df,labels,preds):
    if labels.is_cuda:
        labels = labels.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()
    pre_df = data_df
    labs = []
    pres = []
    for label,pred in zip(labels,preds):
        labs.append(label)
        pres.append(pred)
    pre_df["labels"] = labs
    pre_df["preds"] = pres
    return pre_df

def f_score(y_pre,y,threshold):
    zeros = torch.zeros_like(y_pre)
    ones = torch.ones_like(y_pre)
    y_pre = torch.where(y_pre >= threshold, ones, y_pre)
    y_pre = torch.where(y_pre < threshold, zeros, y_pre)
    # y_pre[:,0]=1
    mm = y_pre*y
    tp = mm.sum(dim=1)
    fp = y_pre.sum(dim=1)-tp
    fn = y.sum(dim=1)-tp
    pr = tp[y_pre.sum(dim=1)!=0] / (tp[y_pre.sum(dim=1)!=0] + fp[y_pre.sum(dim=1)!=0])
    rc = tp[y.sum(dim=1)!=0] / (tp[y.sum(dim=1)!=0] + fn[y.sum(dim=1)!=0]) 
    return pr,rc




 
    
    