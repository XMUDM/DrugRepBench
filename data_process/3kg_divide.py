import os, random
import json
import pandas as pd
import time
import pickle
import networkx as nx
from utils import read_json, write_json, \
    anchor_score_nx, rank_hard_negatives



save_path_train_data = 'kg_merged/train_data.json'

G_path = 'kg_merged/kg-data.gpickle'
node_attrs_path = 'kg_merged/kg-data-nodes.json'

with open(G_path, "rb") as f:
    G_merged = pickle.load(f)



relation_wanted = {
    'indication': ['indication - indication', 'treats', 'prevents'],
    'contraindication': ['contraindication - contraindication', 'contraindicated for'],
    'off-label use': ['off-label use - off-label use']
}



if os.path.exists(save_path_train_data):
    train_data = read_json(save_path_train_data)
else:
    train_data = {
        'indication': [],
        'contraindication': [],
        'off-label use': []
    }
    pair_type = {
        'disease': ['Disease', 'disease', 'Disease|Phenotype', 'Phenotype|Symptom', 'effect/phenotype'], 
        'drug': ['drug', 'Compound', 'Chemical|Compound', 'Chemical', 'Treatment', 'Chemical|Compound|Target', 'Chemical|Target', 'Chemical|Compound|Salt', 'PharmacologicClass']
    }
    for u, v, attr in G_merged.edges(data=True):
        edge_type_list = attr.get("type")

        for rel_type in relation_wanted:
            for edge_type in edge_type_list:
                if edge_type in relation_wanted[rel_type]:
                    u_type = G_merged.nodes[u]['node_attr']['type']
                    v_type = G_merged.nodes[v]['node_attr']['type']

                    if u_type in pair_type['disease']:
                        train_data[rel_type].append( (u, rel_type, v) )
                    else:
                        train_data[rel_type].append( (v, rel_type, u) )
                    break

    write_json(train_data, save_path_train_data)



degree_T = 30
anchor_score_T = 0.3
sem_anchor_score = {        # 仅需计算 source 节点的
    'indication': [],
    'contraindication': [],
    'off-label use': []
}
anchor_score_avg = {
    'indication': [],
    'contraindication': [],
    'off-label use': []
}
# anchor_score_avg: {'indication': [(259.11816194231227, 18243.55373489391, 0.13896730453967296)], 'contraindication': [(125.67089446840772, 5084.5373221401815, 0.12193468926589002)], 'off-label use': [(88.43817995722667, 7291.371395592579, 0.12186149136868821)]}
# degree_avg: {'indication': [(674.822532998651, 20088, 1)], 'contraindication': [(477.6503410420955, 13291, 1)], 'off-label use': [(260.48720349563047, 7947, 1)]} 
degree_avg = {
    'indication': [],
    'contraindication': [],
    'off-label use': []
}
for task in train_data:
    for idx, edge in enumerate(train_data[task]):
        score = anchor_score_nx(G_merged, edge[0])
        degree = G_merged.degree[edge[0]]
        sem_anchor_score[task].append([idx, score, degree])
    
    anchor_score_avg[task] += [sum(x[1] for x in sem_anchor_score[task]) / len(sem_anchor_score[task]), max([x[1] for x in sem_anchor_score[task]]), min([x[1] for x in sem_anchor_score[task]])]
    degree_avg[task] += [sum(x[2] for x in sem_anchor_score[task]) / len(sem_anchor_score[task]), max([x[2] for x in sem_anchor_score[task]]), min([x[2] for x in sem_anchor_score[task]])]

print('anchor_score_avg:', anchor_score_avg)
print('degree_avg:', degree_avg, '\n\n')

cold_start_data = dict()
for i in range(4):
    cname = f'cold_start_{i+1}'
    cold_start_data[cname] = {
        'indication': [],
        'contraindication': [],
        'off-label use': []
    }
for task in sem_anchor_score:
    for edge_pro in sem_anchor_score[task]:
        if edge_pro[2] <= degree_T and edge_pro[1] < anchor_score_avg[task][0]*anchor_score_T:
            cold_start_data['cold_start_1'][task].append(train_data[task][edge_pro[0]])
        if edge_pro[2] <= degree_T and edge_pro[1] >= anchor_score_avg[task][0]*anchor_score_T:
            cold_start_data['cold_start_2'][task].append(train_data[task][edge_pro[0]])
        if edge_pro[2] > degree_T and edge_pro[1] < anchor_score_avg[task][0]*anchor_score_T:
            cold_start_data['cold_start_3'][task].append(train_data[task][edge_pro[0]])
        if edge_pro[2] > degree_T and edge_pro[1] >= anchor_score_avg[task][0]*anchor_score_T:
            cold_start_data['cold_start_4'][task].append(train_data[task][edge_pro[0]])

for situation in cold_start_data:
    print(situation)
    for task in cold_start_data[situation]:
        print('\t', task, len(cold_start_data[situation][task]))

