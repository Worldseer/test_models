#!/usr/bin/env python

import numpy as np
import pandas as pd
import click as ck
import logging
import math
from collections import deque, Counter
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

    train_data_file = os.path.join(data_path,'train_data.pkl')
    test_data_file = os.path.join(data_path,'test_data.pkl')
    diamond_scores_file = os.path.join(data_path,'test_diamond.res')
    obo_path = os.path.join(data_path,'go.obo')
    go_rels = Ontology(obo_path, with_rels=True)
    
    train_df = pd.read_pickle(train_data_file)
    annotations = train_df['annotations'].values
    annotations = list(map(lambda x: set(x), annotations))

    test_df = pd.read_pickle(test_data_file)
    test_annotations = test_df['annotations'].values
    test_annotations = list(map(lambda x: set(x), test_annotations))
    go_rels.calculate_ic(annotations + test_annotations)

    prot_index = {}
    for i, row in enumerate(train_df.itertuples()):
        prot_index[row.proteins] = i

    
    # BLAST Similarity (Diamond)
    diamond_scores = {}
    with open(diamond_scores_file) as f:
        for line in f:
            it = line.strip().split()
            if it[0] not in diamond_scores:
                diamond_scores[it[0]] = {}
            diamond_scores[it[0]][it[1]] = float(it[2])
    
    blast_preds = []
    for i, row in enumerate(test_df.itertuples()):
        annots = {}
        prot_id = row.proteins
        # BlastKNN
        if prot_id in diamond_scores:
            sim_prots = diamond_scores[prot_id]
            allgos = set()
            total_score = 0.0
            for p_id, score in sim_prots.items():
                allgos |= annotations[prot_index[p_id]]
                total_score += score
            allgos = list(sorted(allgos))
            sim = np.zeros(len(allgos), dtype=np.float32)
            for j, go_id in enumerate(allgos):
                s = 0.0
                for p_id, score in sim_prots.items():
                    if go_id in annotations[prot_index[p_id]]:
                        s += score
                sim[j] = s / total_score
            for go_id, score in zip(allgos, sim):
                annots[go_id] = score
            
        blast_preds.append(annots)
        
    go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])
    
    labels = test_annotations
    labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))
    # print(len(go_set))
    fmax = 0.0
    tmax = 0.0
    smin = 1000.0
    precisions = []
    recalls = []
    for t in range(101):
        threshold = t / 100.0
        preds = []
        for i, row in enumerate(test_df.itertuples()):
            annots = set()
            for go_id, score in blast_preds[i].items():
                if score >= threshold:
                    annots.add(go_id)

            new_annots = set()
            for go_id in annots:
                new_annots |= go_rels.get_anchestors(go_id)
            preds.append(new_annots)
        preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), preds))
        
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