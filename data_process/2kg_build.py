import os, random
import json
import pandas as pd
import time
from collections import defaultdict
import pickle

import networkx as nx
from utils import kg2_process, kg1_process, read_json, write_json

pd.set_option('max_colwidth',200)
pd.set_option('display.max_columns', None)



data_merge = read_json('./data/data_merged.json')

kg1_data = kg1_process('data/example_kg1/nodes.csv', 'data/example_kg1/kg.csv', './data/kg1_processed.json')
kg2_data = kg2_process('data/example_kg2/kg.json', './data/kg2_processed.json')

save_path = './kg_merged/kg-data.gpickle'
save_path2 = './kg_merged/kg-data-nodes.json'

start_time_all = time.time()

kg_data = {
    'kg1': kg1_data,
    'kg2': kg2_data,
}

mid_time = time.time()

def build_fused_graph(data_merge, kg_data):
    kg_key = {
        'kg1': ['gene/protein', 'drug', 'effect/phenotype', 'disease', 'biological_process', 'molecular_function', 'cellular_component', 'pathway', 'anatomy'],   # 'exposure' 不对齐, kg2 里没有
        'kg2': ['Gene', 'Compound', 'Symptom', 'Side Effect', 'Disease', 'Biological Process', 'Molecular Function', 'Cellular Component', 'Pathway', 'Anatomy'],     #['Gene', 'Compound', ['Symptom', 'Side Effect'], 'Disease', 'Biological Process', 'Molecular Function', 'Cellular Component', 'Pathway', 'Anatomy'],       #'Pharmacologic Class'
    }
    G = nx.Graph()
    node_info = dict()
    node_map = dict()
    unmerge_id_abs = 0
    merge_id_abs = 0
    continue_time = 0
    edges_to_add = dict()

    # add node
    for i_type, type_record in enumerate(data_merge):
        type_abs = kg_key['kg1'][i_type]
        for db_source in type_record:
            
            # 1.1 merge node
            merge_record = type_record[db_source]['merge_record']
            for idx in range(len(merge_record['kg1'])):
                merged_id = f"merged_{merge_id_abs}_{type_abs}"
                merge_id_abs += 1
                for kg in ['kg1', 'kg2']:
                    ent = merge_record[kg][idx]
                    if ent is not None:
                        ntype = ent[1]
                        nid_unified = ent[-1]
                        if merged_id not in G:
                            G.add_node(merged_id, node_attr={'type': ntype, 'nid_unified': nid_unified})
                        for name_tup in ent[0]:
                            node_map[(kg, ntype, str(nid_unified), name_tup[0])] = merged_id
                            if merged_id not in node_info:
                                node_info[merged_id] = list()
                            if kg == 'kg1':
                                node_info[merged_id].append({
                                    'kg': kg,
                                    'nid_unified': nid_unified,
                                    'type': ntype,
                                    'name': name_tup[0],
                                    'source': ent[2],
                                    'kg_node_id': name_tup[1]
                                })
                            else:
                                node_info[merged_id].append({
                                    'kg': kg,
                                    'nid_unified': nid_unified,
                                    'type': ntype,
                                    'name': name_tup[0],
                                    'source': ent[2]
                                })

            # 1.2 add unmerged node
            unmerged_record = type_record[db_source]['unmerged_record']
            for kg in ['kg1', 'kg2']:
                for idx in range(len(unmerged_record[kg])):
                    unmerged_id = f"unmerged_{type_abs}_{unmerge_id_abs}"
                    unmerge_id_abs += 1
                    ent = unmerged_record[kg][idx]

                    ntype = ent[1]
                    nid_unified = ent[-1]
                    if unmerged_id not in G:
                        G.add_node(unmerged_id, node_attr={'type': ntype, 'nid_unified': nid_unified})

                    for name_tup in ent[0]:
                        node_map[(kg, ntype, str(nid_unified), name_tup[0])] = unmerged_id

                        if unmerged_id not in node_info:
                            node_info[unmerged_id] = list()
                        if kg == 'kg1':
                            node_info[unmerged_id].append({
                                'kg': kg,
                                'nid_unified': nid_unified,
                                'type': ntype,
                                'name': name_tup[0],
                                'source': ent[2],
                                'kg_node_id': name_tup[1]
                            })
                        else:
                            node_info[unmerged_id].append({
                                'kg': kg,
                                'nid_unified': nid_unified,
                                'type': ntype,
                                'name': name_tup[0],
                                'source': ent[2]
                            })

    for kg in ['kg1','kg2']:
        # 1.3 add other node
        nodes = kg_data[kg][0]
        for ntype in nodes:
            for source_db in nodes[ntype]:
                for nid_unified in nodes[ntype][source_db]:
                    node_list = nodes[ntype][source_db][nid_unified]    # [[name, node_id], ...]

                    mark_l = 0
                    for name in node_list:
                        if not (kg, ntype, str(nid_unified), name[0]) in node_map:
                            node_map[(kg, ntype, str(nid_unified), name[0])] = unmerged_id
                            mark_l += 1
                    if not mark_l:
                        unmerged_id = f"unmerged_{ntype}_{unmerge_id_abs}"
                        unmerge_id_abs += 1

                        if unmerged_id not in G:
                            G.add_node(unmerged_id, node_attr={'type': ntype, 'nid_unified': nid_unified})

                        if unmerged_id not in node_info:
                            node_info[unmerged_id] = list()
                        else:
                            print((kg, ntype, nid_unified), 'in node_map 报错')
                            exit()
                        if kg == 'kg1':
                            node_info[unmerged_id].append({
                                'kg': kg,
                                'nid_unified': nid_unified,
                                'type': ntype,
                                'name': node_list[0][0],
                                'source': source_db,
                                'kg_node_id': node_list[0][1]
                            })
                        else:
                            node_info[unmerged_id].append({
                                'kg': kg,
                                'nid_unified': nid_unified,
                                'type': ntype,
                                'name': node_list[0][0],
                                'source': source_db
                            })

    # 2. add edges
    for kg in ['kg1','kg2']:
        edges = kg_data[kg][1]      # [(source_node_info, relation, target_node_infor), ...]
                                    #   source_node_info: (stype, snid_unified, sname, sid)

        for edge_tuple in edges:
            src_type = edge_tuple[0][0]
            src_nid = edge_tuple[0][1]
            src_name = edge_tuple[0][2]

            relation = edge_tuple[1]
            
            tgt_type = edge_tuple[2][0]
            tgt_nid = edge_tuple[2][1]
            tgt_name = edge_tuple[2][2]

            if (not src_nid or 'nan' == str(src_nid)) and src_name == 'NAN':
                continue_time += 1
                continue
            if (not tgt_nid or 'nan' == str(tgt_nid)) and tgt_name == 'NAN':
                continue_time += 1
                continue

            if not (kg, src_type, str(src_nid), src_name) in node_map:
                continue
            if not (kg, tgt_type, str(tgt_nid), tgt_name) in node_map:
                continue
            src_mid = node_map[(kg, src_type, str(src_nid), src_name)]
            tgt_mid = node_map[(kg, tgt_type, str(tgt_nid), tgt_name)]

            if (src_mid, tgt_mid) in edges_to_add:
                edges_to_add[(src_mid, tgt_mid)].append(relation)
            else:
                edges_to_add[(src_mid, tgt_mid)] = [relation]

    edges_to_add_pro = []
    for node_map_tup, relation in edges_to_add.items():
        edges_to_add_pro.append( (node_map_tup[0], node_map_tup[1], {"type": relation}) )
    G.add_edges_from(edges_to_add_pro)

    return G, node_info



G_merged, node_info = build_fused_graph(data_merge, kg_data)

# nx.write_gpickle(G_merged, save_path)
with open(save_path, "wb") as f:
    pickle.dump(G_merged, f)

write_json(node_info, save_path2)

end_time_all = time.time()
print(f'save kg_merged to {save_path}, save kg_merged node info to {save_path2}')
print('Processing time:', end_time_all-start_time_all, 'second')


