import pandas as pd
import numpy as np
from perm_util import split_by_type, get_permutation_ids


def get_cell_types_tedsim(name, fine=True):
  cell_meta = pd.read_csv(f'data/tedsim/cell_meta_{name}.csv', index_col='Unnamed: 0').set_index('cellID')
  cell_meta.index = 't' + cell_meta.index.astype(str)
  if fine:
    return cell_meta[['color']].rename(columns={'color' : 'type'})

  color_coarse_map = {1:'#F8766D', 2:'#B79F00', 3:'#00BA38', 5:'#619CFF', 6:'#F564E3'}
  cell_meta['color-coarse'] = list(map(lambda x : color_coarse_map[x], cell_meta['cluster']))
  return cell_meta[['color-coarse']].rename(columns={'color-coarse' : 'type'})


# cell_types must be a pd.Series with cell names as index
def get_lognorm_expression_tedsim(name, perm_seed=None, cell_types=None, return_perm_ids=False):
  counts = pd.read_csv(f'data/tedsim/counts_{name}.csv').T
  counts.index = counts.index.str.replace('V', 't')
  counts.index.name = 'cellID'

  if perm_seed is not None:
    labels = counts.index.values
    if cell_types is not None:
      label_subsets = split_by_type(labels, cell_types)
    else:
      label_subsets = [labels]
    perm_ids = get_permutation_ids(labels, label_subsets, seed=perm_seed)
    assert cell_types is None or (cell_types.iloc[perm_ids].values == cell_types.values).all()
    counts.index = labels[perm_ids]

  # L1 normalization on cells to [0, 10000]
  counts = counts / counts.sum(axis=1).values.reshape(-1, 1)
  counts *= 10000
  # log2(1 + x) tranform
  counts = np.log2(1 + counts)
  if return_perm_ids:
    return counts, perm_ids
  return counts


def generate_bmtm_data_from_tree(etree, dim, sigma=1, seed=12345, gene_prefix='g'):
  np.random.seed(seed)
  for node in etree.traverse('levelorder'):
    if node == etree:
      etree.add_features(noise=np.random.normal(size=dim, scale=sigma))
    else:
      node.add_features(noise=node.up.noise + np.random.normal(size=dim, scale=sigma))

  leaves = etree.get_leaves()
  labels = []
  data = np.zeros((len(leaves), dim))
  for i, node in enumerate(leaves):
    labels.append(node.name)
    data[i] = node.noise
  return pd.DataFrame(data, index=labels, columns=[f'{gene_prefix}{n:04d}' for n in range(dim)])
