
"""

import pandas as pd
import hashlib
import gc 

df_train = pd.read_csv('train.csv').fillna("")
df_test = pd.read_csv('test.csv').fillna("")


# Generating a graph of Questions and their neighbors
def generate_qid_graph_table(row):
    hash_key1 = hashlib.md5(row["question1"].encode('utf-8')).hexdigest()
    hash_key2 = hashlib.md5(row["question2"].encode('utf-8')).hexdigest()

    qid_graph.setdefault(hash_key1, []).append(hash_key2)
    qid_graph.setdefault(hash_key2, []).append(hash_key1)


qid_graph = {}
print('Apply to train...')
df_train.apply(generate_qid_graph_table, axis=1)
print('Apply to test...')
df_test.apply(generate_qid_graph_table, axis=1)


def pagerank():
    MAX_ITER = 20
    d = 0.85

    # Initializing -- every node gets a uniform value!
    pagerank_dict = {i: 1 / len(qid_graph) for i in qid_graph}
    num_nodes = len(pagerank_dict)

    for iter in range(0, MAX_ITER):

        for node in qid_graph:
            local_pr = 0

            for neighbor in qid_graph[node]:
                local_pr += pagerank_dict[neighbor] / len(qid_graph[neighbor])

            pagerank_dict[node] = (1 - d) / num_nodes + d * local_pr

    return pagerank_dict

print('Main PR generator...')
pagerank_dict = pagerank()

def get_pagerank_value(row):
    q1 = hashlib.md5(row["question1"].encode('utf-8')).hexdigest()
    q2 = hashlib.md5(row["question2"].encode('utf-8')).hexdigest()
    s = pd.Series({
        "q1_pr": pagerank_dict[q1],
        "q2_pr": pagerank_dict[q2]
    })
    return s

print('Apply to train...')
pagerank_feats_train = df_train.apply(get_pagerank_value, axis=1)
print('Writing train...')
pagerank_feats_train.to_csv("pagerank_train.csv", index=False)
del df_train
gc.collect()
print('Apply to test...')
pagerank_feats_test = df_test.apply(get_pagerank_value, axis=1)
print('Writing test...')
pagerank_feats_test.to_csv("pagerank_test.csv", index=False)
