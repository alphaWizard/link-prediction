from itertools import combinations 
import numpy as np
import networkx as nx
import math
# from igraph import *


def apply(graph, func):
    non_edges = nx.non_edges(graph)
    return ((u, v, func(u, v)) for u, v in non_edges)

def CN_score(graph,u,v):
        return len(list(w for w in graph[u] if w in graph[v] and w not in (u, v)))

def common_neighbors_score(graph): 
    non_edges = nx.non_edges(graph)
    def score(u,v):
        return len(list(w for w in graph[u] if w in graph[v] and w not in (u, v)))

    return ((u, v, score(u, v)) for u, v in non_edges) 


def AA_score(graph, u, v):
        return sum(1 / math.log(graph.degree(w))for w in nx.common_neighbors(graph, u, v))

def adamic_adar_score(graph):
    non_edges = nx.non_edges(graph)
    def score(u, v):
        return sum(1 / math.log(graph.degree(w))for w in nx.common_neighbors(graph, u, v))

    return ((u, v, score(u, v)) for u, v in non_edges) 

def PA_score(graph, u, v):
    return (graph.degree(u) * graph.degree(v))


def preferential_attachment_score(graph):
    non_edges = nx.non_edges(graph)
    return ((u, v, graph.degree(u) * graph.degree(v)) for u, v in non_edges)


def RA_score(graph, u, v):
        return sum(1 / graph.degree(w) for w in nx.common_neighbors(graph, u, v))

def resource_allocation_score(graph):
    non_edges = nx.non_edges(graph)
    def score(u, v):
        return sum(1 / graph.degree(w) for w in nx.common_neighbors(graph, u, v))
    return apply(graph, score, non_edges) 


def JC_score(graph, u, v):
        union_size = len(set(graph[u]) | set(graph[v]))
        if union_size == 0:
            return 0
        else:
            return len(list(nx.common_neighbors(graph, u, v))) / union_size

def jaccard_coefficient_score(graph):
    non_edges = nx.non_edges(graph)
    def score(u,v):
        union_size = len(set(graph[u]) | set(graph[v]))
        if union_size == 0:
            return 0
        else:
            return len(list(nx.common_neighbors(graph, u, v))) / union_size

    return apply(graph,score,non_edges)    


def friends_measure(graph, u, v):  #heuristic for katz
    score = 0
    for x in graph[u]:
        for y in graph[v]:
            if x == y:
                score = score + 1
            elif graph.has_edge(x,y):
                score = score + 1
    return score


def katz_score(graph,beta=0.004):
    non_edges = nx.non_edges(graph)
    A = nx.to_numpy_matrix(graph)
    # print(A)
    w, v = np.linalg.eigh(A)
    lambda1 = max([abs(x) for x in w])   # beta should be less than 1/lambda1
    # print(1/lambda1)
    if beta >= 1/lambda1 :
        raise ValueError('beta should be less than 1/lambda, lambda being the eigenvalue with largest magnitude')
    I = np.eye(A.shape[0])
    S = np.linalg.inv(I - beta * A) - I
    return S


def RP_score(graph, u, v):
    return rooted_pagerank_score(graph,u)[v] + rooted_pagerank_score(graph,v)[u]


def rooted_pagerank_score(graph, root_node, alpha = 0.85, max_iter = 100):
    if not graph.is_directed():
        D = graph.to_directed()
    else:
         D = graph
    H = nx.stochastic_graph(D)
    n = len(graph.nodes())
    H = nx.to_numpy_matrix(H).transpose()
    x = np.full((n,1),1.0 / n)
    v = np.full((n,1),0.0)
    nodes = D.nodes()
    index = 0
    for node in nodes:
        # print(node)
        # print(root_node)
        # print(index)
        if node == root_node:
            # print(1-alpha)
            v[index][0] = 1 - alpha
            break
        index = index +  1
    # print(v)

    for _ in range(max_iter):
        x = np.add(alpha*np.dot(H,x),v) # x = alpha*H*x + (1-alpha)*v

    x = np.squeeze(np.asarray(x))
    x = x /	np.sum(x)
    return dict(zip(graph.nodes(), x))



