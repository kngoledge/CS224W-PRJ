import pandas as pd
import numpy as np
from os import path
import timeit
import networkx as nx
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.community.community_utils import is_partition

def main():

  # Column name
  col_name = "ALGORITHM_cmty"

  # Load data
  if path.exists("../data/cmty_nodes.csv"):
    node_upload = "../data/cmty_nodes.csv"
  elif path.exists("../data/nodes.csv"):
    node_upload = "../data/nodes.csv"
  else:
    print("NO NODES TO UPLOAD!")
    assert(False)
  pd_nodes = pd.read_csv(node_upload, sep='\t', index_col=0)

  # Data in nice form
  headers = list(pd_nodes.columns)
  nodes = np.asarray(pd_nodes)

  # Aggregate file names
  model_names = ["GAT","GCN","GraphSage"]
  npy_names = ["../data/"+x+"_node_embeddings.npy" for x in model_names]

  model_cmtys = []
  model_time = []
  for i in range(len(npy_names)):

    # Load embeddings
    embeddings = np.load(npy_names[i])
    print(embeddings.shape)

    # Generate node_mapping for clutsers
    start = timeit.default_timer()
    ##########################################
    # CODE HERE to cluster embeddings and creating node_mapping #
    # node_mapping can either be dictionary or array #
    ##########################################

    node_mapping = np.zeros(len(nodes)).astype(int)

    ##########################################
    stop = timeit.default_timer()
    model_time.append(stop - start)

    # Convert node_mapping to cmtys and node_to_cmty array
    #num_cmtys = len(set(node_mapping.values()))
    num_cmtys = len(set(node_mapping))
    cmtys = [[] for _ in range(num_cmtys)]
    node_to_cmty = np.zeros(len(node_mapping)).astype(int)
    for j in range(len(node_to_cmty)):
      node_to_cmty[j] = node_mapping[j]
      cmtys[node_mapping[j]].append(j)
    model_cmtys.append(cmtys)

    # Add communities to nodes
    pd_nodes[model_names[i]+"_"+col_name] = node_to_cmty
    pd_nodes.to_csv("../data/cmty_nodes.csv", sep='\t')

  print("Creating Graph")
  # Load social network accordingly
  edges = pd.read_csv("../data/edges.csv", sep='\t', index_col=0)
  edges = np.asarray(edges).astype(int)
  G = nx.Graph()
  G.add_nodes_from(range(nodes.shape[0]))
  G.add_edges_from(list(map(tuple, edges)))


  print("Calculating modularity")

  for i in range(len(model_names)):
    assert(is_partition(G, model_cmtys[i]))
    modul = modularity(G, model_cmtys[i])


    print("Results from "+model_names[i]+" ALGORITHM:")
    print("Modularity:",modul)
    print("Number of clusters:",len(model_cmtys[i]))
    print("Time elapsed:",model_time[i])


if __name__ == '__main__':
    main()
