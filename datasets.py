import numpy as np
import networkx as nx
import pandas as pd
import random
import functools
from similarities import *

def memoize(function):
    from functools import wraps

    memo = {}

    @wraps(function)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv
    return wrapper

def load_restaurant_dataset():
    path = 'dataset_ubicomp2013/dataset_ubicomp2013_checkins.txt'
#     lines = (line.decode('utf-8') for line in path)
    infile = open(path, 'r')
    a = set()
    b = set()
    edges = []
    for line in infile:
        s=line.strip().split(None)
        u=-1*int(s.pop(0)) -10
        v=int(s.pop(0))
        a.add(u)
        b.add(v)
        edges.append((u,v))
    top_nodes = {}
    bottom_nodes = {}
    count = 0 
    for x in a:
        top_nodes[x] = count
        count = count + 1
    count  = 0    
    for y in b:
        bottom_nodes[y] = count
        count  = count + 1
    
    A = np.zeros((len(a),len(b)))
    for edge in edges:
        e1 = top_nodes[edge[0]]
        e2 = bottom_nodes[edge[1]]
        A[e1, e2] = 1
    
    A = np.dot(A,A.T)
#     print(A[:35,:35])
    for i in range(0,A.shape[0]):  #making numpy matrix undirected graph type
        for j in range(0,A.shape[1]):
            if i == j :
                A[i,j] = 0
            else:
                if A[i,j] > 0:
                    A[i,j] = 1
                    
    G=nx.from_numpy_matrix(A)
    return G


def load_blog_dataset():
    path = 'BlogCatalog-dataset/data/edges.csv'
    G = nx.Graph()
    edges = pd.read_csv(path, sep=',', header=None)
    G.add_edges_from(edges.values)
    return G
    ## to add for this dataset


def test_restaurant_dataset():
    graph = load_restaurant_dataset()

    nodes = list(graph.nodes)
    # print(nx.info(graph))
    non_edges = list(nx.non_edges(graph))
    edges = list(nx.edges(graph))
    # print((non_edges))
    m = len(edges)

    # print(JC_score(graph, 1, 2))
    # print(m)
    marked_non_edges = random.sample(non_edges, m)
    # print(len(marked_non_edges))
    is_edge = [ 1 ] * m + [ 0 ] * m
        # print(len(is_edge))
    df_dict = {}
    df_dict['is_edge'] = is_edge

    df_dict['node1'] = [ x[0] for x in edges ] + [ x[0] for x in marked_non_edges]
    df_dict['node2'] = [ x[1] for x in edges ] + [ x[1] for x in marked_non_edges]

    # # df = pd.DataFrame(df_dict)
    # # print(df)
    cn_scores = [ CN_score(graph,u,v) for (u,v) in edges ] + [ CN_score(graph,u,v) for (u,v) in marked_non_edges ]
    df_dict['CN_score'] = cn_scores
    print("Added CN scores....")

    aa_scores = [ AA_score(graph,u,v) for (u,v) in edges ] + [ AA_score(graph,u,v) for (u,v) in marked_non_edges ]
    df_dict['AA_score'] = aa_scores
    print("Added AA scores....")

    pa_scores = [ PA_score(graph,u,v) for (u,v) in edges ] + [ PA_score(graph,u,v) for (u,v) in marked_non_edges ]
    df_dict['PA_score'] = pa_scores
    print("Added PA scores....")

    ra_scores = [ RA_score(graph,u,v) for (u,v) in edges ] + [ RA_score(graph,u,v) for (u,v) in marked_non_edges ]
    df_dict['RA_score'] = ra_scores
    print("Added RA scores....")

    jc_scores = [ JC_score(graph,u,v) for (u,v) in edges ] + [ JC_score(graph,u,v) for (u,v) in marked_non_edges ]
    df_dict['JC_score'] = jc_scores
    print("Added JC scores....")

    @functools.lru_cache(maxsize=None)
    def RP_calc(u):
        return rooted_pagerank_score(graph,u)
    
    def RP_compute(u,v):
        return RP_calc(u)[v] + RP_calc(v)[u]

    # print(RP_compute(1,2))
    # print(RP_compute(1,3))
    # print(RP_compute(1,4))
    # for hj in range(2,10):
    #     print(RP_compute(1,hj))

    # for hj in range(2,10):
    #     print(RP_compute(111,hj))
    # return

    rp_scores = [ RP_compute(u, v) for (u,v) in edges ] + [ RP_compute(u,v) for (u,v) in marked_non_edges ]
    df_dict['RP_score'] = rp_scores
    print("Added RP scores....")

    # df = pd.DataFrame(df_dict)
    # print(df)
    katz_matrix = katz_score(graph)
    # print(katz_matrix.shape)
    # print(nodes)
    # katz_scores = []
    # for u,v in edges:
    #     print(u, v)
    #     print(nodes.index(u))
    #     print(nodes.index(v))
    #     katz_scores.append(katz_matrix[nodes.index(u), nodes.index(v)])
    katz_scores = [ katz_matrix[nodes.index(u), nodes.index(v)] for (u,v) in edges ] + [ katz_matrix[nodes.index(u), nodes.index(v)] for (u,v) in marked_non_edges ]
    df_dict['KATZ_score'] = katz_scores
    print("Added KATZ scores....")


    friends_measures = [ friends_measure(graph,u,v) for (u,v) in edges ] + [ friends_measure(graph,u,v) for (u,v) in marked_non_edges ]
    df_dict['friends_measure'] = friends_measures
    print("Added friends measure scores....")

    df = pd.DataFrame(df_dict)
    print(df)








    # print(marked_non_edges)
    # train_edges = edges + marked_non_edges


def test_blog_dataset():
    graph = load_blog_dataset()
    print(nx.info(graph))
    return 




if __name__ == "__main__":
    
    # # print(np.shape(katz_score(graph)))
    # print(katz_score(graph)[1,3])
    test_restaurant_dataset()
    # test_blog_dataset()
    # print(graph.edges())



