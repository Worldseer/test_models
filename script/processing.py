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

    def has_term(self, term_id):# Returns whether the term exists
        return term_id in self.ont

    def get_term(self, term_id):#values returned for the term
        if self.has_term(term_id):
            return self.ont[term_id]
        return None

    def calculate_ic(self, annots):#informance content calculation
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items(): 
        #If there is no parent term, ic is the number of occurrences of this go term
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
                ont[t_id] = ont[term_id]
        for term_id in list(ont.keys()):
            if ont[term_id]['is_obsolete']:#delete obsolete GO terms
                del ont[term_id]
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)
        return ont,obsoletegolist


    def get_anchestors(self, term_id):# Return all ancestral terms
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
        #namespace takes the values BP, MF and CC, this function can retrieve the go_id of all data belonging to a namespace
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):#returns the namespace to which term_id belongs
        return self.ont[term_id]['namespace']
    
    def get_term_set(self, term_id):# Return term and its subterms
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
                if prot_id != '':# Read a new ID value and add the last one to the list if prot_id has data
                    proteins.append(prot_id)
                    accessions.append(prot_ac)
                    sequences.append(seq)
                    annotations.append(annots)
                    interpros.append(ipros)
                    orgs.append(org)
                prot_id = items[1]
                annots = list()
                ipros = list()
                seq = ''
            elif items[0] == 'AC' and len(items) > 1:#AC: access login number, one annotation may have multiple login numbers
                prot_ac = items[1]
            elif items[0] == 'OX' and len(items) > 1:# Classification identifier, data source, e.g. human
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
                seq = next(f).strip().replace(' ', '')
                while True:
                    sq = next(f).strip().replace(' ', '')
                    if sq == '//':
                        break
                    else:
                        seq += sq

        # Add the last data to the list
        proteins.append(prot_id)
        accessions.append(prot_ac)
        sequences.append(seq)
        annotations.append(annots)
        interpros.append(ipros)
        orgs.append(org)
    return proteins, accessions, sequences, annotations, interpros, orgs 

#processing_data removed data with no experimental evidence and propagation annotated
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
        index.append(i)
        annotations.append(annots)
    df = df.iloc[index]
    df = df.reset_index(drop=True)#Update the index of df
    df['exp_annotations'] = annotations# Add another column to df to store GO terms, no evidence code

    prop_annotations = []
    for i, row in df.iterrows():
        # Propagate annotations
        annot_set = set()
        annots = row['exp_annotations']
        for go_id in annots:
            annot_set |= go.get_anchestors(go_id)
        annots = list(annot_set)
        prop_annotations.append(annots)
    df['prop_annotations'] = prop_annotations

    cafa_target = []
    for i, row in enumerate(df.itertuples()):
        if is_cafa_target(row.orgs):
            cafa_target.append(True)
        else:
            cafa_target.append(False)
    df['cafa_target'] = cafa_target# Add a column indicating whether the data is a classification target for cafa
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


# Remove go terms with less data than min_count, divide train_test by train_percent
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
            #Count the number of each GO term in all data
            #in order to remove those terms with insufficient data
    res = {}
    #cnt includes the GO term and its occurrences. Example: 'GO:0010558': 836
    for key, val in cnt.items():
        if val >= int(min_count):
            ont = key.split(':')[0]
            if ont not in res:
                res[ont] = []
            res[ont].append(key)
    terms = []
    for key, val in res.items():
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
 
def split_train_valid(train_pickle,split_rate):
    df = pd.read_pickle(train_pickle)
    n = len(df)
    index = np.arange(n)#Get the index of each data for shuffle
    train_n = int(n * split_rate)# Calculate the amount of training data
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_df = df.iloc[index[:train_n]]
    valid_df = df.iloc[index[train_n:]]
    return train_df, valid_df
    
# count the number of protein data present in a namespace for a go term
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
    
    
#returns all protein data with its annotations based on a list of go terms
#its labels need to be obtained from term_num_df
def data_under_go_list(term_num_df,data_frame):
    go_list = term_num_df.goterm.values.flatten()
    indexset = set()
    data_frame = data_frame.reset_index(drop=True)
    for index,item in data_frame.iterrows():
        for goid in item.prop_annotations:              
            if goid in go_list:
                indexset.add(index)
    return data_frame.iloc[list(indexset),:].reset_index(drop=True)       
