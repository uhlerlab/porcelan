import pandas as pd
import numpy as np


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
