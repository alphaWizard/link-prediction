import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
 

###  Some graph helpful methods###########
def create_graph():
	graph = nx.Graph()
	return graph

def create_digraph():
	di = nx.DiGraph()
	return di


def join(graph,user):
	graph.add_node(user)


def connect(graph,user1,user2):
	if not graph.has_node(user1):
		join(graph,user1)
	if not graph.has_node(user2):
		join(graph,user2)	
	if not graph.has_edge(user1,user2):
		graph.add_edge(user1,user2)


def draw_graph(graph):
	if len(graph.edges()) > 50:
		return
	nx.draw(graph , with_labels=True)
	plt.savefig("graph-drawing.pdf")
	plt.show()    

#############################################
# Functions associated with centrality calculations


def power_iteration(A, max_iter = 100):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    x = np.random.rand(A.shape[1],1)

    for _ in range(max_iter):
        # calculate the matrix-by-vector product Ax
        x1 = np.dot(A, x)

        # calculate the norm
        x1_norm = np.linalg.norm(x1)

        # re normalize the vector
        x = x1 / x1_norm

    return x

# to perform Summation of any Row:
def sum_of_row(A,a):
    degSum = 0
    # Summing all columns in the row
    for column in range(A.shape[1]):
        degSum += A[a,column]
    return degSum
        
# Calculate Degree Centrality for a particular node:
def degreeCentrality(A,a):
    degSum = sum_of_row(A,a)
    # dividing the sum of the degree with the (n-1) nodes
    result = (degSum/(A.shape[1]-1))
    
    # return a single number.
    return result

def degree_centrality(graph):
	A = nx.to_numpy_matrix(graph)
	degreeCentralityValues = [degreeCentrality(A,a) for a in range(A.shape[1])]
	return dict(zip(graph.nodes(), degreeCentralityValues))


def breadth_first_search_distance(graph,source):
    seen={}                  # level (number of hops) when seen in BFS
    level=0                  # the current level
    nextlevel={source:1}  # dict of nodes to check at next level
    while nextlevel:
        thislevel=nextlevel  # advance to next level
        nextlevel={}         # and start a new list (fringe)
        for v in thislevel:
            if v not in seen:
                seen[v]=level # set the level of vertex v
                nextlevel.update(graph[v]) # add neighbors of v
        level=level+1
    return seen  # return all path lengths as dictionary    


def closeness_centrality(graph, u=None, normalized=True):

    if u is None:
        nodes = graph.nodes()
    else:
        nodes = [u]
    closeness_centrality = {}
    for n in nodes:
        sp = breadth_first_search_distance(graph,n)
        totsp = sum(sp.values())
        if totsp > 0.0 and len(graph) > 1:
            closeness_centrality[n] = (len(sp)-1.0) / totsp
            # normalize to number of nodes-1 in connected part
            if normalized:
                s = (len(sp)-1.0) / ( len(graph) - 1 )
                closeness_centrality[n] *= s
        else:
            closeness_centrality[n] = 0.0
    if u is not None:
        return closeness_centrality[u]
    else:
        return closeness_centrality


def single_source_shortest_path(G, s):
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)    # sigma[v]=0 for v in G
    D = {}
    sigma[s] = 1.0
    D[s] = 0
    Q = [s]
    while Q:   # use BFS to find shortest paths
        v = Q.pop(0)
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in G[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:   # this is a shortest path, count paths
                sigma[w] += sigmav
                P[w].append(v)  # predecessors
    return S, P, sigma



def rescale(betweenness, n, normalized, directed=False):
    if normalized is True:
        if n <= 2:
            scale = None  # no normalization b=0 for all nodes
        else:
            scale = 1.0 / ((n - 1) * (n - 2))
    else:  # rescale by 2 for undirected graphs
        if not directed:
            scale = 1.0 / 2.0
        else:
            scale = None
    if scale is not None:
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness


def accumulate(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1.0 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w]
    return betweenness


def betweenness_centrality(graph, normalized = True):
	betweenness = dict.fromkeys(graph, 0.0)  # b[v]=0 for v in G

	nodes = graph
	for s in nodes:
		S, P, sigma = single_source_shortest_path(graph, s)
		betweenness = accumulate(betweenness, S, P, sigma, s)
    # rescaling
	betweenness = rescale(betweenness, len(graph), normalized=normalized,directed=graph.is_directed())
	return betweenness



def eigenvector_centrality(graph, max_iter = 100):
	A = nx.to_numpy_matrix(graph)
	x = np.random.rand(A.shape[1],1)
	for _ in range(max_iter):
		x = np.dot(A, x)  # x = A*x
		x = x /	np.linalg.norm(x)

	x = np.squeeze(np.asarray(x))
	return dict(zip(graph.nodes(), x))



def katz_centrality(graph,alpha = 0.1, beta = 1.0, max_iter = 100):
	A = nx.to_numpy_matrix(graph)
	x = np.random.rand(A.shape[1],1)
	beta_vector = np.full((A.shape[1],1),beta)
	for _ in range(max_iter):
		x = np.add(alpha*np.dot(A,x), beta_vector) # x = alpha*A*x + beta

	x = np.squeeze(np.asarray(x))
	return dict(zip(graph.nodes(), x/np.linalg.norm(x)))



def Google_Matrix(A, alpha):
    N = A.shape[0]
    v = np.ones(N)
    
    # Calculate the degree of each node
    KT = np.dot(A.T, v)

    # Normalize the columns
    for i in range(N):
        A.T[i] = A.T[i]/KT[i]
    
    # Add random links
    S = np.ones((N, N))/N
    G = (1-alpha)*A + alpha*S

    return G


def pagerank(graph,alpha = 0.85, max_iter = 100):
	if not graph.is_directed():
		D = graph.to_directed()
	else:
		D = graph	
	H = nx.stochastic_graph(D)
	n = len(graph.nodes())
	H = nx.to_numpy_matrix(H).transpose()
	x = np.full((n,1),1.0 / n)
	v = np.full((n,1),(1-alpha) / n)

	for _ in range(max_iter):
		x = np.add(alpha*np.dot(H,x),v) # x = alpha*H*x + (1-alpha)*v

	x = np.squeeze(np.asarray(x))
	x = x /	np.sum(x)
	return dict(zip(graph.nodes(), x))

def hits(graph,max_iter = 100):
	A = nx.to_numpy_matrix(graph)
	n = graph.number_of_nodes()
	x = np.random.rand(n,1)  #authority vector
	y = np.random.rand(n,1)  #hub vector
	M = A.T*A  #authority matrix
	N = A*A.T  #hub matrix
	for _ in range(max_iter):
		x = np.dot(M, x)  
		y = np.dot(N, y)
		x = x /	np.sum(x)
		y = y /	np.sum(y)

	x = np.squeeze(np.asarray(x))
	y = np.squeeze(np.asarray(y))
	return (dict(zip(graph.nodes(), x)) , dict(zip(graph.nodes(), y))) #returning both dictionary
