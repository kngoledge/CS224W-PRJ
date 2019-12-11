import fastcluster
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hac

a = np.array([[0.1,   2.5],
              [1.5,   .4 ],
              [0.3,   1  ],
              [1  ,   .8 ],
              [0.5,   0  ],
              [0  ,   0.5],
              [0.5,   0.5],
              [2.7,   2  ],
              [2.2,   3.1],
              [3  ,   2  ],
              [3.2,   1.3]])

method = 'ward'

# Aggregate file names
model_names = ["GAT","GCN","GraphSage"]
npy_names = ["../data/"+x+"_node_embeddings_with_features.npy" for x in model_names]

model_cmtys = []
model_time = []
for i in range(len(npy_names)):

  # Load embeddings
  embeddings = np.load(npy_names[i])
  print(embeddings.shape)

  plt.figure()
  z = fastcluster.linkage(embeddings, method=method)

  # Plotting
  plt.plot(range(1, len(z)+1), z[::-1, 2])
  knee = np.diff(z[::-1, 2], 2)
  plt.plot(range(2, len(z)), knee)

  num_clust1 = knee.argmax() + 2
  print(num_clust1)

  plt.text(num_clust1, z[::-1, 2][num_clust1-1], 'possible\n<- knee point')

  node_embeddings = hac.fcluster(z, num_clust1, 'maxclust')

  m = '\n(method: {})'.format(method)
  plt.title('Screeplot{}'.format(m))
  plt.xlabel(xlabel='partition')
  plt.ylabel('{}\ncluster distance'.format(m))

  plt.savefig('../figures/'+model_names[i]+'_agglomerative.png')
