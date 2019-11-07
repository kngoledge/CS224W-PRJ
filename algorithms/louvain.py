import snap
import pandas as pd
import numpy as np
from os import path

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

  # Louvain Algorithm from snap.py
  CmtyV = snap.TCnComV()
  undirected = snap.ConvertGraph(snap.PUNGraph, social_network)
  snap.DelSelfEdges(undirected)
  modularity = snap.CommunityCNM(undirected, CmtyV)
  #for Cmty in CmtyV:
  #  print("Community Size:",len(Cmty))
  print("Results from Clauset-Newman-Moore:")
  print("Modularity:",modularity)
  print("Number of clusters:",len(CmtyV))

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
