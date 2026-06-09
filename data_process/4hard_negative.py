import random
import json
import pickle
import tqdm
import numpy as np
from utils import read_json, read_txt, write_json, rank_hard_negatives, compute_struct_score, semantic_hard_negative_score, mechanism_diff



save_path_hard_nega = 'kg_merged/hard_nega.json'

embs_path = 'data/embeddings.json'              # use your own embbeding for kg nodes
drug_mechset_path = 'data/drug_mechset.json'    # use your own drug mechanism set
node_info_path = './kg_merged/kg-data-nodes.json'
kg_path = './kg_merged/kg-data.gpickle'

test_data = 'kg_merged/test_example.txt'

test_data = read_txt(test_data)
test_data = test_data.split('\n')

relation_wanted = {
    'indication': ['indication - indication', 'treats', 'prevents'],
    'contraindication': ['contraindication - contraindication', 'contraindicated for'],
    'off-label use': ['off-label use - off-label use']
}
test_pro = {
    'indication': [],
    'contraindication': [],
    'off-label use': []
}
for triplet in test_data:
    disease, relation, drug = triplet.split('\t')
    for label in test_pro:
        if relation in relation_wanted[label]:
            test_pro[label].append(triplet)
            break

node_info = read_json(node_info_path)
drug_mechset = read_json(drug_mechset_path)
embs = read_json(embs_path)

drug_type = ['drug', 'Compound', 'Chemical|Compound', 'Chemical', 'Treatment', 'Chemical|Compound|Target', 'Chemical|Target', 'Chemical|Compound|Salt', 'PharmacologicClass']
drugs_set = set()
embs_pro = dict()
for drug_id in embs:

    embs_pro[drug_id] = np.array(embs[drug_id], dtype=float)
    for tup_n in node_info[drug_id]:
        if tup_n['type'] in drug_type:
            drugs_set.add(drug_id)
            break

with open(kg_path, "rb") as f:
    G_merged = pickle.load(f)


test_plus = dict()
for label in test_pro:
    if label == 'indication' or label == 'contraindication':
        test_plus[label] = random.sample(test_pro[label], 600)
    else:
        test_plus[label] = test_pro[label]

test_max = dict()
for label in test_plus:
    test_max[label] = dict()

    for triplet in test_plus[label]:
        disease, relation, drug = triplet.split('\t')

        if disease not in test_max[label]:
            test_max[label][disease] = {
                'in_test': dict(),
                'out_test': []
            }

            disease_neighbs = list(G_merged.neighbors(disease))
            for neighb in disease_neighbs:
                rel_type = G_merged.get_edge_data(disease, neighb)['type']
                for rel in rel_type:
                    if rel in relation_wanted[label]:
                        test_max[label][disease]['out_test'].append(neighb)
                        break

        test_max[label][disease]['in_test'][drug] = {
            'relation': relation
        }

max_deg = max(dict(G_merged.degree()).values())
for d in drug_mechset:
    drug_mechset[d] = set(drug_mechset[d])

for label in test_max:
    for disease in tqdm.tqdm(test_max[label], desc=label):
        in_test = set(list(test_max[label][disease]['in_test'].keys()))
        out_test = set(test_max[label][disease]['out_test'])
        
        drugs_test = drugs_set - in_test
        drugs_test = list(drugs_test - out_test)

        for drug_posi in test_max[label][disease]['in_test']:
            sorted_candidates_struct, sorted_candidates_semi, sorted_candidates_mix = rank_hard_negatives(max_deg, G_merged, drugs_test, drug_posi, embs_pro, drug_mechset)
            test_max[label][disease]['in_test'][drug_posi]['hard-nega-candidates-struct'] = sorted_candidates_struct
            test_max[label][disease]['in_test'][drug_posi]['hard-nega-candidates-semi'] = sorted_candidates_semi
            test_max[label][disease]['in_test'][drug_posi]['hard-nega-candidates-mix'] = sorted_candidates_mix

write_json(test_max, save_path_hard_nega)


