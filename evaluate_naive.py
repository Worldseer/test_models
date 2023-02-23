#!/usr/bin/env python


import numpy as np
import pandas as pd
import click as ck
import logging
import math
import os
from script.utils import FUNC_DICT, Ontology, NAMESPACES
# from matplotlib import pyplot as plt

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--data-path', '-p', default='../data_2016',
    help='Data file with training features')
@ck.option(
    '--ont', '-o', default='mf',
    help='GO subontology (bp, mf, cc)')
def main(data_path, ont):

    train_data_file = os.path.join(data_path,'train_data_train.pkl')
    test_data_file = os.path.join(data_path,'test_data.pkl')
    # diamond_scores_file = os.path.join(data_path,'test_diamond.res')
    obo_path = os.path.join(data_path,'go.obo')
    go_rels = Ontology(obo_path, with_rels=True)
    
    train_df = pd.read_pickle(train_data_file)
    annotations = train_df['annotations'].values
    annotations = list(map(lambda x: set(x), annotations))

    test_df = pd.read_pickle(test_data_file)
    test_annotations = test_df['annotations'].values
    test_annotations = list(map(lambda x: set(x), test_annotations))
    go_rels.calculate_ic(annotations + test_annotations)

    go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])
    
    annotations = list(map(lambda x: set(filter(lambda y: y in go_set, x)), annotations))
    
    cnt = Counter()
    max_n = 0
    for x in annotations:
        cnt.update(x)
    print(cnt.most_common(10))
    max_n = cnt.most_common(1)[0][1]
    print(max_n)
    scores = {}
    for go_id, n in cnt.items():
        score = n / max_n
        scores[go_id] = score

    prot_index = {}
    for i, row in enumerate(train_df.itertuples()):
        prot_index[row.proteins] = i


    labels = test_annotations
    labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))
    print(len(go_set))
    fmax = 0.0
    tmax = 0.0
    smin = 1000.0
    precisions = []
    recalls = []
    for t in range(101):
        threshold = t / 100.0
        preds = []
        annots = set()
        for go_id, score in scores.items():
            if score >= threshold:
                annots.add(go_id)
            # new_annots = set() 
            # for go_id in annots:
            #     new_annots |= go_rels.get_anchestors(go_id)
            # new_annots = set(filter(lambda y: y in go_set, new_annots))
        for i, row in enumerate(test_df.itertuples()):
            preds.append(annots.copy())
        
        fscore, prec, rec, s = evaluate_annotations(go_rels, labels, preds)
        precisions.append(prec)
        recalls.append(rec)
        print(f'Fscore: {fscore}, Precision: {prec}, Recall: {rec} Smin: {s}, threshold: {threshold}')
        if fmax < fscore:
            fmax = fscore
            tmax = threshold
        if smin > s:
            smin = s
    print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}')
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    print(f'AUPR: {aupr:0.3f}')


def evaluate_annotations(go, real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    p_total= 0
    ru = 0.0
    mi = 0.0
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = real_annots[i].intersection(pred_annots[i])
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        for go_id in fp:
            mi += go.get_ic(go_id)
        for go_id in fn:
            ru += go.get_ic(go_id)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
    ru /= total
    mi /= total
    r /= total
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s


if __name__ == '__main__':
    main()