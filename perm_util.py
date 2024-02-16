import pandas as pd
import numpy as np
from ete3 import Tree


def split_by_type(names_to_keep, cell_types):
  df = pd.DataFrame(names_to_keep, columns=['name']).set_index('name').join(cell_types).reset_index()
  assert (names_to_keep == df['name']).all()
  subsets = []
  for cell_type in df['type'].unique():
    subsets.append(df[df['type'] == cell_type]['name'].values.tolist())
  return subsets


# permutes labels within the subsets and returns then maps permutation to 
# original label order
def get_permutation_ids(labels_in_order, label_subsets, seed):
  label_sorter = np.argsort(labels_in_order)
  label_sorter_inv = np.argsort(label_sorter)
  label_subset_order = np.concatenate(label_subsets)
  label_subset_order_sorter = np.argsort(label_subset_order)
  label_subset_order_sorter_inv = np.argsort(label_subset_order_sorter)

  subset_to_canon_order = label_subset_order_sorter[label_sorter_inv]
  assert np.all(label_subset_order[subset_to_canon_order] == labels_in_order)
  canon_to_subset_order = label_sorter[label_subset_order_sorter_inv]
  assert np.all(label_subset_order == labels_in_order[canon_to_subset_order])
  assert np.all(subset_to_canon_order[canon_to_subset_order] == np.arange(len(labels_in_order)))

  # permute subsets
  np.random.seed(seed)
  subset_perm = np.arange(len(labels_in_order))
  left = 0
  for subset in label_subsets:
    np.random.shuffle(subset_perm[left:left+len(subset)])
    left += len(subset)

  canon_perm = canon_to_subset_order[subset_perm][subset_to_canon_order]
  return canon_perm


def get_cluster_ids(tree, labels, depth=2):
  cluster_roots = [tree.seed_node]
  for i in range(depth):
    cluster_children = []
    for root in cluster_roots:
      if root.is_leaf():
        cluster_children.append(root)
      else:
        cluster_children.extend(list(root.child_node_iter()))
    cluster_roots = cluster_children
  labels_to_ids = {} 
  cluster_id = 1
  for root in cluster_roots:
    for node in root.leaf_nodes():
      labels_to_ids[node.taxon.label] = cluster_id
    cluster_id += 1
  return np.array([node.label for node in cluster_roots]), np.array([labels_to_ids[x] for x in labels])


def subset_labels(subsets, labels):
  subset_labels = np.zeros(len(labels), dtype=int)
  for i, subset in enumerate(subsets):
    subset_labels += (i+1) * np.isin(labels, subset)
  return subset_labels


def extract_subsets_with_depth(tree, depth, include_labels, return_ids=False):
  if isinstance(tree, str): 
    tree = Tree(tree, quoted_node_names=True, format=1)
  else:
    tree = tree.copy()
  subsets = []
  arrived_at_root = False
  while not arrived_at_root:
    to_prune = []
    for node in tree.iter_search_nodes():
      if node.get_farthest_leaf()[1] == depth:
        # post order includes node but not leaves
        sub_labels = np.array([child.name for child in node.iter_search_nodes()])
        sub_labels = sub_labels[np.isin(sub_labels, include_labels)]
        if len(sub_labels) > 0:
          subsets.append(sub_labels)
        to_prune.append(node)
    if len(to_prune) > 0:
      for node in to_prune:
        if node.is_root():
          arrived_at_root = True 
          break
        node.detach()
    else:
      assert depth > 0
      depth -= 1  # try shallower subtrees
  
  assert sum(map(len, subsets)) == len(include_labels)
  if not return_ids:
    return subsets
  
  id_subsets = []
  for sub in subsets:
    isub = []
    for label in sub:
      isub.append(np.argmax(include_labels == label).item())
    id_subsets.append(isub)
  return id_subsets
