import os
import json
import pandas as pd
import time

#设置value的显示长度为200，默认为50
pd.set_option('max_colwidth',200)
#显示所有列，把行显示设置成最大
pd.set_option('display.max_columns', None)

def write_txt(text, output_path):
    with open(output_path, 'w', encoding='utf-8') as fw:
        fw.write(text)

def read_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
    return data

def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def write_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as fw:
        json.dump(data, fw, ensure_ascii=False, indent=4)



output_path = './data/data_merged.json'
kg1 = './data/example_kg1/example_kg1.json'         # type -> source -> ID -> list: [name, node_index in kg]
kg2 = './data/example_kg2/example_kg2.json'         # type -> source -> ID -> list: [name]

kg1 = read_json(kg1)
kg2 = read_json(kg2)

# select the shared node categories
kg1_key = ['gene/protein', 'drug', 'effect/phenotype', 'disease', 'biological_process', 'molecular_function', 'cellular_component', 'pathway', 'anatomy']
kg2_key = ['Gene', 'Compound', ['Symptom', 'Side Effect'], 'Disease', 'Biological Process', 'Molecular Function', 'Cellular Component', 'Pathway', 'Anatomy']

start_time_all = time.time()
merge_record = []
for i, key1 in enumerate(kg1_key):
    start_time = time.time()
    key2 = kg2_key[i]
    type_record = dict()

    for s1 in kg1[key1]:
        merge_info = {
            'node_num_before_merge': {
                'kg1': len(kg1[key1][s1]),
                'kg2': 0
            }, 
            'merged_num': 0,
            'merge_record': {       # [[name, node_id in kg], type, source, ID]
                'kg1': [],  
                'kg2': []
            },
            'unmerged_record': {    # [[name, node_id in kg], type, source, ID]
                'kg1': [],  
                'kg2': []
            }
        }
        ent1_set = set()
        ent2_set = set()
        ent_duplicate_merged_set = dict()

        if type(key2) is list:
            for k in key2:
                for s2 in kg2[k]:
                    merge_info['node_num_before_merge']['kg2'] += len(kg2[k][s2])
        else:
            for s2 in kg2[key2]:
                merge_info['node_num_before_merge']['kg2'] += len(kg2[key2][s2])
    
        for nid1 in kg1[key1][s1]:
            mark_finish = False
            # kg2
            if type(key2) is list:
                for k in key2:
                    for s2 in kg2[k]:
                        for nid2 in kg2[k][s2]:
                            if nid2 in ent2_set:
                                continue
                            if nid2 == nid1:
                                ent1_set.add(nid1)
                                ent2_set.add(nid2)
                                merge_info['merge_record']['kg1'].append( [kg1[key1][s1][nid1], key1, s1, nid1] )
                                merge_info['merge_record']['kg2'].append( [kg2[k][s2][nid2], k, s2, nid2] )
                                merge_info['merged_num'] += 1
                                mark_finish = True
                                break
                        if mark_finish:
                            break
                    if mark_finish:
                        break
            else:
                for s2 in kg2[key2]:
                    for nid2 in kg2[key2][s2]:
                        if nid2 in ent2_set:
                            continue
                        if nid2 == nid1:
                            ent1_set.add(nid1)
                            ent2_set.add(nid2)
                            merge_info['merge_record']['kg1'].append( [kg1[key1][s1][nid1], key1, s1, nid1] )
                            merge_info['merge_record']['kg2'].append( [kg2[key2][s2][nid2], key2, s2, nid2] )
                            merge_info['merged_num'] += 1
                            mark_finish = True
                            break
                    if mark_finish:
                        break

            if not nid1 in ent1_set:
                merge_info['unmerged_record']['kg1'].append( [kg1[key1][s1][nid1], key1, s1, nid1] )
        
        if type(key2) is list:
            for k in key2:
                for s2 in kg2[k]:
                    for nid2 in kg2[k][s2]:
                        if not nid2 in ent2_set:
                            merge_info['unmerged_record']['kg2'].append( [kg2[k][s2][nid2], k, s2, nid2] )
        else:
            for s2 in kg2[key2]:
                for nid2 in kg2[key2][s2]:
                    if not nid2 in ent2_set: 
                        merge_info['unmerged_record']['kg2'].append( [kg2[key2][s2][nid2], key2, s2, nid2] )

        type_record[s1] = merge_info
    if type_record:
        merge_record.append(type_record)
    end_time = time.time()
    print(f'time about kg1  {key1} emerged with kg2 {key2}:', end_time-start_time, 'second\n')
end_time_all = time.time()
write_json(merge_record, output_path)
print(f'save merge_record to {output_path}')
print('All processing time:', end_time_all-start_time_all, 'second')
