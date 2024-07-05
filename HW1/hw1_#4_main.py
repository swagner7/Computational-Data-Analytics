import numpy as np
import sys
import os

import scipy.stats

np.set_printoptions(threshold=sys.maxsize)
from scipy.sparse import csgraph
from scipy import sparse
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def read_edges_file():
	filepath = os.path.join(os.getcwd(), 'edges.txt')
	edges = []
	with open(filepath) as txt:
		for line in txt.readlines(): # for each line in txt file
			edges.append(line.split()) # split line and add to list storage

	edges = np.array(edges).astype(int) # convert list to array
	return edges


def read_nodes_file():
	filepath = os.path.join(os.getcwd(), 'nodes.txt')
	parties = []
	with open(filepath) as txt:
		for line in txt.readlines(): # for each line in txt file
			parties.append(line.split('\t')[2]) # split line and add to list storage

	parties = np.array(parties).astype(int) # convert list to array
	return parties


def remove_unused_nodes(edges, nodes):
	keep_nodes = []
	removed_nodes = []
	for i in range(1, nodes.size+1):
		if i in edges:
			keep_nodes.append(nodes[i-1])
		else:
			removed_nodes.append(nodes[i-1])

	nodes = np.array(keep_nodes)
	return nodes


def build_adjacency_matrix(edges, parties):
	# build adjacency graph
	graph = {}
	for edge in edges:
		u, v = edge
		if u not in graph:
			graph[u] = []
		if v not in graph:
			graph[v] = []
		graph[u].append(v)
		graph[v].append(u)

	# build adjacency matrix from graph
	keys = sorted(graph.keys())
	size = len(keys)

	A = [[0] * size for i in range(size)]

	# We iterate over the key:value entries in the dictionary first,
	# then we iterate over the elements within the value
	for a, b in [(keys.index(a), keys.index(b)) for a, row in graph.items() for b in row]:
		# Use 1 to represent if there's an edge
		# Use 2 to represent when node meets itself in the matrix (A -> A)
		A[a][b] = 2 if (a == b) else 1

	return np.array(A)


def build_laplacian_matrix(A):
	L = csgraph.laplacian(A)

	return L


def eigen_decomp(matrix, k):
	lambd, v = np.linalg.eig(matrix)
	idx = lambd.argsort()
	v = v[:, idx]
	v = v[:, 0:k].real

	return(lambd, v)


def kmeans(k, Z):

	kmeans = KMeans(n_clusters=k, n_init='auto').fit(Z)
	idx = kmeans.labels_
	centroids = kmeans.cluster_centers_

	return idx, centroids


def calc_mismatch(k, idx, parties):

	mismatches = 0

	for i in range(0, k):

		cluster_members = parties[idx == i]
		majority = scipy.stats.mode(cluster_members, keepdims = True)[0][0]
		print('Cluster: ', i)
		print('Majority: ', majority, '\n')

		cluster_mismatches = 0
		for j in cluster_members:
			if j != majority:
				cluster_mismatches += 1
				mismatches += 1


	mismatch_pct = mismatches/parties.shape[0]
	return mismatch_pct


mismatches = []
ks = []
for k in range(2, 200):
	print('Starting spectral clustering...\n')
	print('Reading txt files...\n')
	edges = read_edges_file()
	parties = read_nodes_file()
	parties = remove_unused_nodes(edges, parties)

	print('Building adjacency matrix...\n')
	A = build_adjacency_matrix(edges, parties)

	print('Building Laplacian matrix...\n')
	L = build_laplacian_matrix(A)

	print('Performing eigen decomposition...\n')
	lambd, v = eigen_decomp(L, k)

	print('Running k-means clustering of eigenvectors...\n')
	idx, centroids = kmeans(k, v)

	print('Calculating mismatch rate...')
	mismatch = calc_mismatch(k, idx, parties)
	print('Overall mismatch Rate: ', mismatch)

	mismatches.append(mismatch)
	ks.append(k)

plt.plot(ks, mismatches)
plt.title('Mismatch Percentage vs Number of Clusters')
plt.xlabel('Clusters')
plt.ylabel('Mismatch %')
plt.show()

