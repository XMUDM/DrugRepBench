import networkx as nx
import math, os, json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd



def kg1_process(nodes_path, kg_path, processed_path):
    """
    return: 
    - nodes (dict): 
        node_type -> node_source -> node_id -> list: [name, node_index]
    - edges (list): 
        [(source_node_info, relation, target_node_infor), ...]
          source_node_info: (stype, snid_unified, sname, sid)
    """

    if os.path.exists(processed_path):
        data = read_json(processed_path)
        return data['nodes'], data['edges']

    nodes = pd.read_csv(nodes_path)
    data_classified = dict()            # node_type -> node_source -> node_id -> list: [name, node_index]
    for row in nodes.itertuples():
        if not row.node_type in data_classified:
            data_classified[row.node_type] = dict()
        if not row.node_source in data_classified[row.node_type]:
            data_classified[row.node_type][row.node_source] = dict()

        if row.node_source == 'HPO':
            nid_unified = row.node_id.rjust(7, "0")
            nid_unified = "HP:" + nid_unified
        elif row.node_source == 'MONDO':
            nid_unified = row.node_id.rjust(7, "0")
            nid_unified = "MONDO:" + nid_unified
        elif row.node_source == 'GO':
            nid_unified = row.node_id.rjust(7, "0")
            nid_unified = "GO:" + nid_unified
        elif row.node_source == 'UBERON':
            nid_unified = row.node_id.rjust(7, "0")
            nid_unified = "UBERON:" + nid_unified
        else:
            nid_unified = row.node_id

        if not nid_unified in data_classified[row.node_type][row.node_source]:
            data_classified[row.node_type][row.node_source][nid_unified] = []

        if pd.isna(row.node_name):
            name = 'NAN'
        else:
            name = row.node_name

        data_classified[row.node_type][row.node_source][nid_unified].append([name, row.node_index])

    kg1 = pd.read_csv(kg_path)
    edge_record = list()
    for row in kg1.itertuples():
        stype = row.x_type
        sid = row.x_index
        if row.x_source == 'HPO':
            snid_unified = str(row.x_id).rjust(7, "0")
            snid_unified = "HP:" + snid_unified
        elif row.x_source == 'MONDO':
            snid_unified = str(row.x_id).rjust(7, "0")
            snid_unified = "MONDO:" + snid_unified
        elif row.x_source == 'GO':
            snid_unified = str(row.x_id).rjust(7, "0")
            snid_unified = "GO:" + snid_unified
        elif row.x_source == 'UBERON':
            snid_unified = str(row.x_id).rjust(7, "0")
            snid_unified = "UBERON:" + snid_unified
        else:
            snid_unified = str(row.x_id)
        if pd.isna(row.x_name):
            sname = 'NAN'
        else:
            sname = row.x_name

        ttype = row.y_type
        tid = row.y_index
        if row.y_source == 'HPO':
            tnid_unified = str(row.y_id).rjust(7, "0")
            tnid_unified = "HP:" + tnid_unified
        elif row.y_source == 'MONDO':
            tnid_unified = str(row.y_id).rjust(7, "0")
            tnid_unified = "MONDO:" + tnid_unified
        elif row.y_source == 'GO':
            tnid_unified = str(row.y_id).rjust(7, "0")
            tnid_unified = "GO:" + tnid_unified
        elif row.y_source == 'UBERON':
            tnid_unified = str(row.y_id).rjust(7, "0")
            tnid_unified = "UBERON:" + tnid_unified
        else:
            tnid_unified = str(row.y_id)
        if pd.isna(row.y_name):
            tname = 'NAN'
        else:
            tname = row.y_name

        relation = row.relation + ' - ' + row.display_relation

        edge_record.append( ((stype, snid_unified, sname, sid), relation, (ttype, tnid_unified, tname, tid)) )      # (source_node_info, relation, target_node_infor)

    output = {
        'nodes': data_classified,
        'edges': edge_record
    }

    write_json(output, processed_path)

    return data_classified, edge_record



def kg2_process(source_path, processed_path):
    """
    return: 
    - nodes (dict):
    - edges (list): 
        [(source_node_info, relation, target_node_infor), ...]
          source_node_info: (stype, snid_unified, sname, sid)
    """

    if os.path.exists(processed_path):
        data = read_json(processed_path)
        return data['nodes'], data['edges']
    
    data_2 = read_json(source_path)
    nodes = data_2['nodes']
    edges = data_2['edges']

    data_classified = dict()        # kind -> ['data']['source'] -> identifier -> list: [name]
    nodes_pro = dict()              # type -> id -> name
    for node_tuple in nodes:
        if not node_tuple['kind'] in data_classified:
            data_classified[node_tuple['kind']] = dict()
        if not node_tuple['data']['source'] in data_classified[node_tuple['kind']]:
            data_classified[node_tuple['kind']][node_tuple['data']['source']] = dict()
        if not node_tuple['identifier'] in data_classified[node_tuple['kind']][node_tuple['data']['source']]:
            data_classified[node_tuple['kind']][node_tuple['data']['source']][node_tuple['identifier']] = []
        name = node_tuple['name']

        data_classified[node_tuple['kind']][node_tuple['data']['source']][node_tuple['identifier']].append([name])

        if not node_tuple['kind'] in nodes_pro:
            nodes_pro[node_tuple['kind']] = dict()
        if not node_tuple['identifier'] in nodes_pro[node_tuple['kind']]:
            nodes_pro[node_tuple['kind']][node_tuple['identifier']] = node_tuple['name']
    
    edge_record = []
    for edge in edges:
        stype = edge['source_id'][0]
        snid_unified = edge['source_id'][1]
        sname = nodes_pro[stype][snid_unified]

        ttype = edge['target_id'][0]
        tnid_unified = edge['target_id'][1]
        tname = nodes_pro[ttype][tnid_unified]

        relation = edge['kind']

        edge_record.append( ((stype, snid_unified, sname), relation, (ttype, tnid_unified, tname)) )

    output = {
        'nodes': data_classified,
        'edges': edge_record
    }

    write_json(output, processed_path)

    return data_classified, edge_record



















# cold-start
def anchor_score_nx(G, disease_node, weights=None):
    """
    计算疾病节点的锚点分数 (基于 NetworkX 图对象)
    
    参数:
    G: networkx.Graph
        包含疾病节点及其邻居的图
    disease_node: 节点标识
        要计算的疾病节点 (必须在图中)
    weights: dict
        类型权重映射，默认值:
        {
            "gene": 3,
            "pathway": 2,
            "phenotype": 1.5,
            "literature": 1,
            "other": 1
        }
    
    节点属性要求:
    - 每个邻居节点需要有 "type" 属性 (如 "gene", "pathway", "phenotype", "literature")
    
    返回:
    float: 锚点分数
    """
    if weights is None:
        weights = {
            "gene": 3,
            "pathway": 2,
            "phenotype": 1.5,
            "literature": 1,
            "other": 1
        }
    
    score = 0.0
    for n in G.neighbors(disease_node):
        node_type = G.nodes[n].get("node_attr")['type']
        degree = G.degree[n]
        
        w = weights.get(node_type, weights["other"])
        penalty = 1 / math.log(1 + degree)
        
        score += w * penalty
    
    return score



# hard negative
# 1. struct
def compute_struct_score(G, candidate_drug, positive_drug, alpha=0.5, beta=0.5):
    """
    计算候选药物的结构难度分数
    
    参数:
    G : networkx.Graph
        图结构，节点可以是药物或疾病
    candidate_drug : 节点
        候选药物节点
    positive_drug : 节点
        正样本药物节点
    alpha, beta : float
        权重参数
    
    返回:
    float : 结构难度分数
    """
    # 度数归一化部分
    deg_candidate = G.degree(candidate_drug)
    max_deg = max(dict(G.degree()).values())
    deg_score = deg_candidate / max_deg if max_deg > 0 else 0
    
    # 邻居交集部分
    neighbors_candidate = set(G.neighbors(candidate_drug))
    neighbors_positive = set(G.neighbors(positive_drug))
    intersection_size = len(neighbors_candidate & neighbors_positive)
    overlap_score = intersection_size / len(neighbors_positive) if len(neighbors_positive) > 0 else 0
    
    # 综合得分
    struct_score = alpha * deg_score + beta * overlap_score
    return struct_score



# 2. sematic
def semantic_hard_negative_score(drug_emb, pos_emb, mechanism_diff, gamma=0.5):
    """
    计算语义难度分数 S_sem(d)

    参数:
    - drug_emb: np.array, 候选药物的语义嵌入向量
    - pos_emb: np.array, 正样本药物的语义嵌入向量
    - mechanism_diff: float, 候选药物与正样本的机制差异分数 (越大表示机制差异越大)
    - gamma: float, 权重系数，用于调节机制差异的影响

    返回:
    - score: float, 语义难度分数
    """
    # 计算余弦相似度
    cos_sim = cosine_similarity(drug_emb.reshape(1, -1), pos_emb.reshape(1, -1))[0][0]
    
    # 计算语义难度分数
    score = cos_sim - gamma * mechanism_diff
    return score



# 衡量药物作用机制差异的函数
def mechanism_diff(drug_mechanism, pos_mechanism):
    """
    计算药物机制差异分数 (MechanismDiff)

    参数:
    - drug_mechanism: set, 候选药物的机制集合 (如靶点、通路、ATC分类等)
    - pos_mechanism: set, 正样本药物的机制集合
    返回:
    - diff: float, 差异分数 (0 表示完全相同, 1 表示完全不同)
    """
    if not drug_mechanism and not pos_mechanism:
        return 0.0
    
    intersection = len(drug_mechanism.intersection(pos_mechanism))
    union = len(drug_mechanism.union(pos_mechanism))
    
    if union == 0:
        return 1.0  # 没有任何机制信息时，认为差异最大
    
    jaccard_similarity = intersection / union
    diff = 1 - jaccard_similarity
    return diff



# 难负样本场景批量排序函数
def rank_hard_negatives(G, candidate_drugs, positive_drug, drug_embs, drug_mechset, gamma=0.5, top_k=5):
    """
    批量计算候选药物的语义难度分数，并返回 Top-k 难负样本
    
    参数:
    - G : networkx.Graph, 图结构，节点可以是药物或疾病
    - candidate_drugs : 节点, 候选药物节点
    - positive_drug : 节点
    - drug_embs: dict，每个候选药物的 emb, {id: np.array,  ...}
    - drug_mechset: dict, 每个候选药物的 "mechanism" 集合, {id: set, ...}
    - gamma: float, 机制差异权重
    - top_k: int, 返回前 k 个难负样本
    
    返回:
    - sorted_candidates: list of dict，按分数排序的前 k 个候选药物
    """

    pos_emb = drug_embs[positive_drug]
    pos_mech = drug_mechset[positive_drug]

    results = dict()
    for drug in candidate_drugs:
        struct_score = compute_struct_score(G, drug, positive_drug)
        sem_score = semantic_hard_negative_score(drug_embs[drug], pos_emb, mechanism_diff(drug_mechset[drug], pos_mech), gamma)
        
        score = struct_score + sem_score
        results[drug] = score
    
    # 按分数降序排序
    sorted_candidates = sorted(candidate_drugs, key=lambda x: results[x], reverse=True)
    return sorted_candidates[:top_k]



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


