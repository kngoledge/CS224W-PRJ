import snap
import pandas as pd
import numpy as np
from os import path
import scipy.sparse
import scipy.sparse.linalg

def data2dag(data, num_nodes):
  dag = snap.TNGraph.New()
  for i in range(num_nodes):
    dag.AddNode(i)

  for i in range(data.shape[0]):
    dag.AddEdge(int(data[i][0]), int(data[i][1]))
  FOut = snap.TFOut("../data/youtube.graph")
  dag.Save(FOut)
  return dag

def main():

  # Load data
  nodes = pd.read_csv("../data/nodes.csv", sep='\t', index_col=0)

  # Data in nice form
  headers = list(nodes.columns)
  nodes = np.asarray(nodes)

  # Load social network accordingly
  if path.exists("../data/adjacency.npy"):
    A = np.load("../data/adjacency.npy")
  else:
    edges = pd.read_csv("../data/edges.csv", sep='\t', index_col=0)
    edges = np.asarray(edges).astype(int)
    A = np.zeros((nodes.shape[0], nodes.shape[0]))
    A[edges[:,0],edges[:,1]] = 1
    A[edges[:,1],edges[:,0]] = 1
    np.save("../data/adjacency", A)
    print("edges:",edges.shape[0])
  csgraph = scipy.sparse.csgraph.csgraph_from_dense(A)

  laplacian = scipy.sparse.csgraph.laplacian(csgraph)
  laplacian_dense = scipy.sparse.csr_matrix.todense(laplacian)

  eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(laplacian_dense, k=12, which="SM")
  print(eigenvalues)


if __name__ == '__main__':
    main()
