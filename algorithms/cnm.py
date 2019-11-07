import snap
import pandas as pd
import numpy as np
from os import path
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
  if path.exists("../data/youtube.graph"):
    FIn = snap.TFIn("../data/youtube.graph")
    social_network = snap.TNGraph.Load(FIn)
  else:
    edges = pd.read_csv("../data/edges.csv", sep='\t', index_col=0)
    edges = np.asarray(edges).astype(int)
    social_network = data2dag(edges, nodes.shape[0])

  # Check for self edges
  for e in social_network.Edges():
    if e.GetSrcNId() == e.GetDstNId():
      print("Self Loop Found:",e.GetSrcNId())

  # CNM Algorithm from snap.py
  print("Computing CNM")
  start = timeit.default_timer()
  CmtyV = snap.TCnComV()
  undirected = snap.ConvertGraph(snap.PUNGraph, social_network)
  snap.DelSelfEdges(undirected)
  modularity = snap.CommunityCNM(undirected, CmtyV)
  stop = timeit.default_timer()
  node_to_cmty = np.zeros(nodes.shape[0])
  cmty_sizes = np.zeros(len(CmtyV))
  for i in range(len(CmtyV)):
    for node in CmtyV[i]:
      node_to_cmty[node] = i
    cmty_sizes[i] = len(CmtyV[i])
  print("Results from Clauset-Newman-Moore:")
  print("Modularity:",modularity)
  print("Number of clusters:",len(CmtyV))
  print("Time elapsed:",stop - start)


  # Fun category stuff to do
  '''
  upload_col = headers.index('category')
  categories = set()
  for i in range(nodes.shape[0]):
    categories.add(nodes[i][upload_col])
  idx_to_categories = list(categories)
  print("Number of categories:",len(idx_to_categories))
  categories_to_idx = dict()
  for i in range(len(idx_to_categories)):
    categories_to_idx[idx_to_categories[i]] = i

  # Communities and categories
  cmty_category_count = np.zeros((len(CmtyV),len(idx_to_categories)))
  for i in range(nodes.shape[0]):
    cmty_category_count[int(node_to_cmty[i]),categories_to_idx[nodes[i][upload_col]]] += 1
  cmty_category_count = cmty_category_count/cmty_sizes[:,np.newaxis]
  '''


  # Create graphs per category
  '''
  plt.figure()
  for i in range(len(idx_to_categories)):
    if (str(idx_to_categories[i]) != "nan") and (idx_to_categories[i] != " UNA "):
      plt.plot(sorted(cmty_category_count[:,i], reverse=True), label=idx_to_categories[i])
  plt.title("Category Proportions in Clusters")
  plt.xlabel("Cluster")
  plt.ylabel("Proportion")
  plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
  plt.savefig("../figures/category_proportions_clusters.png", bbox_inches="tight")
  '''
  '''
  for i in range(cmty_category_count.shape[0]):
    top_category = np.argmax(cmty_category_count[i])
    print("Community "+str(i)+": "+str(idx_to_categories[top_category])+",",cmty_category_count[i][top_category])
  '''





  '''
  This algorithm is not working!
  CmtyV = snap.TCnComV()
  modularity = snap.CommunityGirvanNewman(undirected, CmtyV)
  #for Cmty in CmtyV:
  #  print("Community Size:",len(Cmty))
  print("Modularity using Girvan-Newman algorithm is %f" % modularity)
  '''


if __name__ == '__main__':
    main()
