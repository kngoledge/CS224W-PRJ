import pandas as pd
from os import path

def main():

  # Column name
  cols_to_delete = ["GCN_test_cmty","GraphSage_test_cmty","GAT_test_cmty"]

  # Load data
  if path.exists("../data/cmty_nodes.csv"):
    node_upload = "../data/cmty_nodes.csv"
  else:
    print("NO NODES TO UPLOAD!")
    assert(False)
  pd_nodes = pd.read_csv(node_upload, sep='\t', index_col=0)
  headers = set(pd_nodes.columns)
  for col in cols_to_delete:
    assert(col in headers)

  # Add communities to nodes
  pd_nodes[model_names[i]+"_"+col_name] = node_to_cmty
  pd_nodes.to_csv("../data/cmty_nodes.csv", sep='\t')


if __name__ == '__main__':
    main()
