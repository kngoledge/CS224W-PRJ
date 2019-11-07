import snap
import pandas as pd
import numpy as np
from os import path
import networkx as nx
import pylab as plt
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import pygraphviz as pgv

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

  # Find number of unique uploaders
  upload_col = headers.index('category')
  categories = set()
  for i in range(nodes.shape[0]):
    categories.add(nodes[i][upload_col])
  idx_to_categories = list(categories)
  print("Number of categories:",len(idx_to_categories))
  categories_to_idx = dict()
  for i in range(len(idx_to_categories)):
    categories_to_idx[idx_to_categories[i]] = i

  # Create edge weights based on categories
  edge_weights = dict()
  for e in social_network.Edges():
    src = nodes[int(e.GetSrcNId())][upload_col]
    dst = nodes[int(e.GetDstNId())][upload_col]
    if src != dst:
      edge_weights[(src,dst)] = edge_weights.get((src,dst),0) + 1
      edge_weights[(dst,src)] = edge_weights.get((dst,src),0) + 1

  # Color Scheme
  # Colors based on: https://www.graphviz.org/doc/info/colors.html
  bucket_bounds = [500, 750, 1000, 2000, 5000]
  colors = ['lightskyblue','lightslateblue','blue','darkviolet','midnightblue']

  # Draw networkx graph
  G = nx.Graph()
  for cat in idx_to_categories:
    G.add_node(cat)
  to_keep = set()
  all_weights = set()
  for edge in edge_weights:
    bucket = -1
    while (bucket+1 < len(bucket_bounds)) and (edge_weights[edge] > bucket_bounds[bucket+1]):
      bucket += 1
    if bucket >= 0:
      G.add_edge(edge[0],edge[1],penwidth=np.power(edge_weights[edge], 0.25), color = colors[bucket])
      to_keep.add(edge[0])
      to_keep.add(edge[1])
      all_weights.add(edge_weights[edge])

  all_weights = list(all_weights)
  print(sorted(all_weights))

  # Remove unused nodes
  for cat in idx_to_categories:
    if cat not in to_keep:
      G.remove_node(cat)

  # Plot graph
  A = to_agraph(G)
  A.layout('dot')
  A.draw('../figures/category.png')




if __name__ == '__main__':
    main()
