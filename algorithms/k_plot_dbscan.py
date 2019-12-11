import numpy as np
import matplotlib.pyplot as plt

def k_distances2(x, k):
  dim0 = x.shape[0]
  dim1 = x.shape[1]
  p=-2*x.dot(x.T)+np.sum(x**2, axis=1).T+ np.repeat(np.sum(x**2, axis=1),dim0,axis=0).reshape(dim0,dim0)
  p = np.sqrt(p)
  p.sort(axis=1)
  p=p[:,:k]
  pm= p.flatten()
  pm= np.sort(pm)
  return p, pm

model_names = ["GAT","GCN","GraphSage"]
npy_names = ["../data/"+x+"_node_embeddings.npy" for x in model_names]
for i in range(len(npy_names)):
  embeddings = np.load(npy_names[i])
  print(embeddings.shape)
  m, m2= k_distances2(embeddings, 4)
  plt.plot(m2)
  plt.ylabel("k-distances")
  plt.grid(True)
  plt.savefig("../figures/"+model_names[i]+"_dbscan_k_plot.png")
