import numpy as np
import pandas as pd
import gzip
import math
import pickle
from collections import deque, Counter

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

CAFA_TARGETS = set([
    '287', '3702', '4577', '6239', '7227', '7955', '9606', '9823', '10090',
    '10116', '44689', '83333', '99287', '226900', '243273', '284812', '559292'])

def is_cafa_target(org):
    return org in CAFA_TARGETS

def is_exp_code(code):
    return code in EXP_CODES

class Ontology(object):
    
    def __init__(self, filename='input/go.obo', with_rels=False):
        self.ont, self.obsoletegolist = self.loadgo(filename, with_rels)
        self.ic = None

    def has_term(self, term_id):#返回该术语是否存在
        return term_id in self.ont

    def get_term(self, term_id):#返回该术语的values
        if self.has_term(term_id):
            return self.ont[term_id]
        return None

    def calculate_ic(self, annots):#informance content计算：信息内容
        cnt = Counter()
        #collection模块的Counter函数，用来统计词出现的次数，返回一个类似字典的东西
        for x in annots:#x应该是个字典,只包括GOterm如：{'GO:0002134'}
            cnt.update(x)
        self.ic = {}#为字典
        for go_id, n in cnt.items(): 
        #如果没有父术语，ic为此go术语出现的次数
        #如果有父术语则为其所有术语中，出现次数最少的术语
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

    def loadgo(self, filename, with_rels):
        ont = dict()
        obj = None
        obsoletegolist = list()
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()#
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
                    elif l[0] == 'namespace':#BP、MF、CC
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
                        obsoletegolist.append(obj['id'])
            if obj is not None:
                ont[obj['id']] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
            #如果有alt_ids,则添加一个与主id相同数据的alt_ids
                ont[t_id] = ont[term_id]
        for term_id in list(ont.keys()):
            if ont[term_id]['is_obsolete']:#删除过期的GO术语
                del ont[term_id]
        for term_id, val in ont.items():
            if 'children' not in val:
            #避免有的数据没有key=children,首先将所有的数据添加一条为空的key=children
            #其实可以没有这一句
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)
        return ont,obsoletegolist


    def get_anchestors(self, term_id):#返回所有的祖先术语
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)#也可以不用因为term_set为集合，集合无重复数据。因此这一步是为了优化处理时间
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set


    def get_parents(self, term_id):#返回该术语的is_a的术语
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set


    def get_namespace_terms(self, namespace):
    #namespace取值有BP、MF和CC，该函数可以取出属于某个namespace的所有数据的go_id
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):#返回term_id属于的namespace（mf、bp、cc）
        return self.ont[term_id]['namespace']
    
    def get_term_set(self, term_id):#返回term和它的子术语
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

def save_go(go_obo,go_pickle):
    data = Ontology(go_obo)
    with open(go_pickle,'wb') as f:
        pickle.dump(data,f)

def load_sequencedata(go_obo,sequence_gz_data):
    go = Ontology(go_obo)
    proteins = list()
    accessions = list()
    sequences = list()
    annotations = list()
    interpros = list()
    orgs = list()
    with gzip.open(sequence_gz_data, 'rt') as f:
        prot_id = ''
        prot_ac = ''
        seq = ''
        org = ''
        annots = list()
        ipros = list()
        for line in f:
            items = line.strip().split('   ')
            if items[0] == 'ID' and len(items) > 1:
                if prot_id != '':#读到一个新的ID值，如果prot_id有数据则将上次的数据添加到列表中
                    proteins.append(prot_id)
                    accessions.append(prot_ac)
                    sequences.append(seq)
                    annotations.append(annots)
                    interpros.append(ipros)
                    orgs.append(org)
                prot_id = items[1]
                annots = list()#数据类型为列表需要重置，数据类型如果为字符串可直接覆盖
                ipros = list()#
                seq = ''
            elif items[0] == 'AC' and len(items) > 1:#AC：access登录号，一个annotation可能有多个登录号
                prot_ac = items[1]
            elif items[0] == 'OX' and len(items) > 1:#分类标识符，数据来源，如人类
                if items[1].startswith('NCBI_TaxID='):
                    org = items[1][11:]#取出来源ID
                    end = org.find(' ')
                    org = org[:end]
                else:
                    org = ''
            elif items[0] == 'DR' and len(items) > 1:
                items = items[1].split('; ')
                if items[0] == 'GO':
                    go_id = items[1]
                    if go_id not in go.obsoletegolist:
                        code = items[3].split(':')[0]
                        annots.append(go_id + '|' + code)
                if items[0] == 'InterPro':
                    ipro_id = items[1]
                    ipros.append(ipro_id)
            elif items[0] == 'SQ':
                seq = next(f).strip().replace(' ', '')#把空格都去除
                while True:
                    sq = next(f).strip().replace(' ', '')
                    if sq == '//':
                        break
                    else:
                        seq += sq

        #将最后一个数据添加到列表中
        proteins.append(prot_id)
        accessions.append(prot_ac)
        sequences.append(seq)
        annotations.append(annots)
        interpros.append(ipros)
        orgs.append(org)
    return proteins, accessions, sequences, annotations, interpros, orgs 

#processing_data删除了没有实验证据的数据，并进行了传播注释
def data2df(go_obo, sequence_gz_data):
    go = Ontology(go_obo)
    proteins, accessions, sequences, annotations, interpros, orgs = load_sequencedata(go_obo,sequence_gz_data)
    df = pd.DataFrame({
        'proteins': proteins,
        'accessions': accessions,
        'sequences': sequences,
        'annotations': annotations,
        'interpros': interpros,
        'orgs': orgs
    })
    index = []
    annotations = []
    for i, row in enumerate(df.itertuples()):
        annots = []
        for annot in row.annotations:
            go_id, code = annot.split('|')
            if is_exp_code(code):
                annots.append(go_id)
        # Ignore proteins without experimental annotations
        if len(annots) == 0:
            continue
        index.append(i)#将有实验证据的数据的索引存储到列表，没有的通过continue已经跳过
        annotations.append(annots)
    df = df.iloc[index]
    df = df.reset_index(drop=True)#将df的索引更新
    df['exp_annotations'] = annotations#再df中添加一列，存储GO术语，没有证据代码

    prop_annotations = []
    for i, row in df.iterrows():
        # Propagate annotations
        annot_set = set()
        annots = row['exp_annotations']#是包含的GO术语
        for go_id in annots:
            annot_set |= go.get_anchestors(go_id)#进行位运算
        annots = list(annot_set)
        prop_annotations.append(annots)
    df['prop_annotations'] = prop_annotations

    cafa_target = []
    for i, row in enumerate(df.itertuples()):
        if is_cafa_target(row.orgs):
            cafa_target.append(True)
        else:
            cafa_target.append(False)
    df['cafa_target'] = cafa_target#添加一列，表示数据是否时cafa的分类目标
    length = list()
    seqlist = df.sequences.values.flatten()
    for seq in seqlist:
        length.append(len(seq))
    df['length'] = length        
    bp = set()
    bpo = list()
    mf = set()
    mfo = list()
    cc = set()
    cco = list()
    for i,items in df.iterrows():
        for goid in items.exp_annotations:
            if goid in go.ont:
                if go.ont[goid]['namespace'] == 'biological_process' and i not in bp:
                    bpo.append(items)
                    bp.add(i)
                if go.ont[goid]['namespace'] == 'molecular_function' and i not in mf:
                    mfo.append(items)
                    mf.add(i)
                if go.ont[goid]['namespace'] == 'cellular_component' and i not in cc:
                    cco.append(items)
                    cc.add(i)
    bpo = pd.DataFrame(bpo).reset_index(drop=True)
    mfo = pd.DataFrame(mfo).reset_index(drop=True)
    cco = pd.DataFrame(cco).reset_index(drop=True)
    #df.to_pickle(out_pickle)
    return bpo,mfo,cco



def df2go(go_obo, df):
    go = Ontology(go_obo)
    length = list()
    seqlist = df.sequences.values.flatten()
    for seq in seqlist:
        length.append(len(seq))
    df['length'] = length        
    bp = set()
    bpo = list()
    mf = set()
    mfo = list()
    cc = set()
    cco = list()
    for i,items in df.iterrows():
        # if items.cafa_target == True:
        for goid in items.exp_annotations:
            if goid in go.ont:
                if go.ont[goid]['namespace'] == 'biological_process' and i not in bp:
                    bpo.append(items)
                    bp.add(i)
                if go.ont[goid]['namespace'] == 'molecular_function' and i not in mf:
                    mfo.append(items)
                    mf.add(i)
                if go.ont[goid]['namespace'] == 'cellular_component' and i not in cc:
                    cco.append(items)
                cc.add(i)
    bpo = pd.DataFrame(bpo).reset_index(drop=True)
    mfo = pd.DataFrame(mfo).reset_index(drop=True)
    cco = pd.DataFrame(cco).reset_index(drop=True)
    #df.to_pickle(out_pickle)
    return bpo,mfo,cco


#去除数据少于min_count的go术语，按照train_percent进行train_test划分
def split_train_test(go_obo,df_file,
         out_terms_file, train_data_file, test_data_file, min_count, train_percent):
    go = Ontology(go_obo)
    #with_rels=True表示考虑regulate、has_part、part_of关系，都加入到is_a中
    df = pd.read_pickle(df_file)
    print("DATA FILE", len(df))
    cnt = Counter()
    annotations = list()
    for i, row in df.iterrows():
        for term in row['prop_annotations']:
            cnt[term] += 1
            #统计所有数据中每个GO术语的数量
            #以便去除那些数据量不足的术语
    res = {}
    #cnt包括GO术语和其出现的次数例：'GO:0010558': 836
    for key, val in cnt.items():
        if val >= int(min_count):
            ont = key.split(':')[0]
            if ont not in res:
                res[ont] = []
            res[ont].append(key)#res包括一个key：list即"GO"：["GO:0010558","'GO:2001141'"]
    terms = []
    for key, val in res.items():#key为字符串"GO",val为包括GO术语的列表
        print(key, len(val))
        terms += val  
    # Save the list of terms
    terms_df = pd.DataFrame({'terms': terms})
    terms_df.to_pickle(out_terms_file)
    n = len(df)
    # Split train/valid
    index = np.arange(n)
    train_n = int(n * float(train_percent))
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_df = df.iloc[index[:train_n],1:]#去除索引
    train_df = train_df.reset_index(drop=True)
    test_df = df.iloc[index[train_n:],1:]
    test_df = test_df.reset_index(drop=True)

    print('Number of train proteins', len(train_df))
    train_df.to_pickle(train_data_file)

    print('Number of test proteins', len(test_df))
    test_df.to_pickle(test_data_file)

#不会保存验证数据，直接加载进内存    
def split_train_valid(train_pickle,split_rate):
    df = pd.read_pickle(train_pickle)
    n = len(df)#n表示有多少个数据
    index = np.arange(n)#获取每个数据的索引，以便进行shuffle
    train_n = int(n * split_rate)#计算训练数据的数量
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_df = df.iloc[index[:train_n]]
    valid_df = df.iloc[index[train_n:]]
    return train_df, valid_df
    
#统计某个namespace中go术语存在的蛋白质数据数量
def computenum(go_obo,namespce,data_frame):
    go = Ontology(go_obo)
    cnt = Counter()
    for _,item in data_frame.iterrows():
        for goterm in item.prop_annotations:
            if goterm in go.ont:
                if go.get_namespace(goterm) == namespce:
                    cnt[goterm] += 1
    df = pd.DataFrame.from_dict(cnt,orient='index').reset_index()
    df.columns = ['goterm','num']
    df = df.sort_values(by='num',ascending=False,ignore_index=True)
    return df
    
    
#根据go术语列表，返回有其注释的所有蛋白质数据
#其标签需要从term_num_df获得
def data_under_go_list(term_num_df,data_frame):
    go_list = term_num_df.goterm.values.flatten()
    indexset = set()
    data_frame = data_frame.reset_index(drop=True)
    for index,item in data_frame.iterrows():
        for goid in item.prop_annotations:              
            if goid in go_list:
                indexset.add(index)
    return data_frame.iloc[list(indexset),:].reset_index(drop=True)       