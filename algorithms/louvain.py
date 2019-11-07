import snap
import pandas as pd
import numpy as np
from os import path
import networkx as nx
import community
import matplotlib.pyplot as plt
import timeit

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
  edges = pd.read_csv("../data/edges.csv", sep='\t', index_col=0)
  edges = np.asarray(edges).astype(int)
  G = nx.Graph()
  G.add_nodes_from(range(nodes.shape[0]))
  G.add_edges_from(list(map(tuple, edges)))

  #first compute the best partition
  print("Computing Louvain Algorithm")
  start = timeit.default_timer()
  partition = community.best_partition(G)
  stop = timeit.default_timer()

  # Computing modularity
  num_cmtys = len(set(partition.values()))
  num_edges = edges.shape[0]
  cmtys = [[] for _ in range(num_cmtys)]
  for node in partition.keys():
    cmtys[partition[node]].append(node)

  # Load social network accordingly
  if path.exists("../data/youtube.graph"):
    FIn = snap.TFIn("../data/youtube.graph")
    social_network = snap.TNGraph.Load(FIn)
  else:
    social_network = data2dag(edges, nodes.shape[0])

  modularity = 0
  for cmty in cmtys:
    Nodes = snap.TIntV()
    for elem in cmty:
      Nodes.Add(int(elem))
    modularity += snap.GetModularity(social_network, Nodes, num_edges)
  print("Results from Louvain:")
  print("Modularity:",modularity)
  print("Number of clusters:",num_cmtys)
  print("Time elapsed:",stop - start)


  #drawing
  '''
  size = float(len(set(partition.values())))
  pos = nx.spring_layout(G)
  count = 0.
  for com in set(partition.values()) :
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                node_color = str(count / size))

  nx.draw_networkx_edges(G, pos, alpha=0.5)
  plt.show()
  '''


if __name__ == '__main__':
    main()
