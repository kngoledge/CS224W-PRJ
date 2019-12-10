import pandas as pd
import numpy as np
from os import path
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt



def main():

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
  npy_names = ["../data/"+x+"_node_embeddings_with_features.npy" for x in model_names]

  versions = []

  for i in range(len(npy_names)):

    # Load embeddings
    embeddings = np.load(npy_names[i])
    print(embeddings.shape)

    # Generate node_mapping for clutsers
    ##########################################
    # CODE HERE to cluster embeddings and creating node_mapping #
    # node_mapping can either be dictionary or array #
    ##########################################
    # Standardize the data
    X_std = StandardScaler().fit_transform(embeddings)

    sse = []
    list_k = list(range(1, 50))

    for k in list_k:
      print("Running KMeans for k =",k)
      km = KMeans(n_clusters=k)
      km.fit(X_std)
      sse.append(km.inertia_)
    versions.append(sse)

    # Plot sse against k
    plt.figure()
    for j in range(len(versions)):
      plt.plot(list_k, versions[j], '-', label=model_names[j])
    plt.title("Sum of Squared Distances for Various k in KMeans Clustering")
    plt.xlabel('Number of clusters *k*')
    plt.ylabel('Sum of squared distance')
    plt.legend()
    plt.savefig("../figures/find_k_with_features"+str(i)+".png")

    ##########################################


if __name__ == '__main__':
    main()
