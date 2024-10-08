import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import trange, tqdm
from ete3 import Tree

from perm_util import extract_subsets_with_depth, get_permutation_ids
from tree_util import get_tree_dists, get_tree_info


# expression should be a normalized vector
# gene_group_size does not affect result but setting this will save memory by computing the result in multiple steps of the specified size 
def local_autocorrelation(weights, expression, agg_func=torch.mean, gene_group_size=None):
  weight_scale = torch.sqrt(torch.sum(torch.square(weights)))
  if len(expression.shape) == 1:
    return torch.sum(weights * (expression[None] * expression[:, None])) / weight_scale
  else:
    if gene_group_size is None:
      per_gene_score = torch.sum(weights[:, :, None] * (expression[None] * expression[:, None]), axis=(0, 1)) / weight_scale
    else:
      per_gene_scores = []
      for left in range(0, expression.shape[1], gene_group_size):
        per_gene_scores.append(torch.sum(weights[:, :, None] * (expression[None, :, left:left+gene_group_size] * expression[:, None, left:left+gene_group_size]), axis=(0, 1)))
      per_gene_score = torch.cat(per_gene_scores) / weight_scale

    return agg_func(per_gene_score)
  

def get_dist_corr_for_all_genes(tree_dists, expression, label_subset=None):
  if label_subset is not None:
    insubset = expression.index.isin(label_subset)
    tree_dists = tree_dists[insubset][:, insubset]
    expression = expression.iloc[insubset]

  expression_tensor = torch.from_numpy(expression.values)
  expression_dists = torch.sqrt((expression_tensor[None] - expression_tensor[:, None])**2)
  tree_dists = torch.from_numpy(tree_dists)
  scores = []
  for i in range(len(expression.columns)):
    scores.append(torch.corrcoef(torch.vstack((tree_dists.flatten(), expression_dists[:,:,i].flatten())))[0,1].item())
  scores = pd.DataFrame(scores, index=expression.columns, columns=['score'])

  return scores


def get_lac_for_all_genes(tree_dists, expression, label_subset=None, device='cuda:0', 
                          decay_factor=10, gene_group_size=None):
  if label_subset is not None:
    insubset = expression.index.isin(label_subset)
    tree_dists = tree_dists[insubset][:, insubset]
    expression = expression.iloc[insubset]

  norm_expression = torch.tensor((expression.values - expression.values.mean(axis=0))/expression.values.std(axis=0)).float().to(device)
  tree_dists = torch.from_numpy(tree_dists).to(device)
  weights = torch.triu(torch.exp(- torch.square(tree_dists) / decay_factor), diagonal=1)
  scores = local_autocorrelation(weights, norm_expression, agg_func=lambda x: x, gene_group_size=gene_group_size).cpu()
  scores = pd.DataFrame(scores, index=expression.columns, columns=['score'])

  return scores


def get_trip_scores_for_all_genes(apn_lut, expression, label_subset=None, device='cuda:0'):
  if label_subset is not None:
    insubset = expression.index.isin(label_subset)
    apn_lut = apn_lut[insubset][:, insubset][:, :, insubset]
    expression = expression.iloc[insubset]

  expression_tensor = torch.from_numpy(expression.values).to(device)
  apn_triplets = torch.argwhere(apn_lut).to(device)
  scores_ntl = []
  scores_tc = []
  for i in range(len(expression.columns)):
    scores_ntl.append(-F.triplet_margin_with_distance_loss(
      anchor=expression_tensor[apn_triplets[:, 0], i][:, None],
      positive=expression_tensor[apn_triplets[:, 1], i][:, None],
      negative=expression_tensor[apn_triplets[:, 2], i][:, None]).item())

    ap_dist = F.pairwise_distance(expression_tensor[apn_triplets[:, 0], i][:, None],
                                  expression_tensor[apn_triplets[:, 1], i][:, None], keepdim=True)
    an_dist = F.pairwise_distance(expression_tensor[apn_triplets[:, 0], i][:, None],
                                  expression_tensor[apn_triplets[:, 2], i][:, None], keepdim=True)
    scores_tc.append((ap_dist < an_dist).float().mean().item())

  scores = pd.DataFrame(scores_ntl, index=expression.columns, columns=['ntl'])
  scores['tc'] = scores_tc

  return scores


# gene_group_size does not affect result but setting this will save memory by computing the 
# result in multiple steps of the specified size 
def get_lac_for_all_subtrees_all_genes(tree_path, expression_df, decay_factor=10, quoted_node_names=False, 
                                       device='cuda:0', skip_descendents_size=0, skip_tip_distance=0, gene_group_size=None):
  _, edges_ends, _, edge_weights, paths, skipped, leaf0_ancestors = get_tree_info(tree_path, skip_tip_distance=skip_tip_distance,
                                                                          initial_weight=1, labels_in_order=expression_df.index,
                                                                          quoted_node_names=quoted_node_names, skip_descendents_size=skip_descendents_size)
  edge_weights = torch.from_numpy(edge_weights).float().to(device)
  paths = torch.from_numpy(paths).float().to(device)
  skipped = torch.from_numpy(skipped).float().to(device)

  dist = (paths * edge_weights[None, None, :]).sum(axis=-1) + skipped  # add weights of skipped edges
  weights = torch.triu(torch.exp(- torch.square(dist) / decay_factor), diagonal=1)

  scores_subtree = pd.DataFrame(index=edges_ends, columns=expression_df.columns.values.tolist() + ['# nodes'])
  scores_remaining = pd.DataFrame(index=edges_ends, columns=expression_df.columns.values.tolist() + ['# nodes'])

  if gene_group_size is None:
    gene_group_size = len(expression_df.columns)

  for left in range(0, len(expression_df.columns), gene_group_size):
    gene_group = expression_df.columns[left:left+gene_group_size]
    expression = torch.tensor(expression_df[gene_group].values).to(device)

    for k, e in tqdm(enumerate(edges_ends), total=len(edges_ends)):
      if e in leaf0_ancestors:
        other = torch.argwhere(paths[0, :, k] > 0).flatten()
        downstream = torch.argwhere(paths[0, :, k] == 0).flatten()
      else:
        other = torch.argwhere(paths[0, :, k] == 0).flatten()
        downstream = torch.argwhere(paths[0, :, k] > 0).flatten()

      if len(downstream) > 1:
        norm_expression = ((expression[downstream] - expression[downstream].mean(axis=0))/expression[downstream].std(axis=0))
        scores_subtree.loc[e, gene_group] = local_autocorrelation(weights[downstream, :][:, downstream], norm_expression, agg_func=lambda x: x).cpu()
        scores_subtree.loc[e, '# nodes'] = len(downstream)

      if len(other) > 1:
        norm_expression = ((expression[other] - expression[other].mean(axis=0))/expression[other].std(axis=0))
        scores_remaining.loc[e, gene_group] = local_autocorrelation(weights[other, :][:, other], norm_expression, agg_func=lambda x: x).cpu()
        scores_remaining.loc[e, '# nodes'] = len(other)

  return scores_subtree, scores_remaining


def lac_theoretical(mrca_depths, tree_dists, gamma=10):
  weights = np.triu(np.exp(-tree_dists**2/gamma), k=1)
  N = len(mrca_depths)
  V = np.sum(np.diag(mrca_depths)) / (N-1) - np.sum(mrca_depths)/(N * (N-1))
  C = np.sqrt(np.sum(weights**2))

  covs = mrca_depths
  covs = covs - covs.mean(axis=1).reshape(-1, 1) - covs.mean(axis=0).reshape(1, -1) + covs.mean(axis=(0, 1))
  return np.sum(weights * covs) / (C * V)


def lac_theoretical_perm(mrca_depths, tree_dists, subsets, gamma=10):
  weights = np.triu(np.exp(-tree_dists**2/gamma), k=1)
  N = len(mrca_depths)
  V = np.sum(np.diag(mrca_depths)) / (N-1) - np.sum(mrca_depths)/(N * (N-1))
  C = np.sqrt(np.sum(weights**2))

  covs = mrca_depths
  covs = covs - covs.mean(axis=1).reshape(-1, 1) - covs.mean(axis=0).reshape(1, -1) + covs.mean(axis=(0, 1))
  # covs = np.triu(covs, k=1)
  lac = 0
  for p in subsets:
    if len(p) > 1:
      lac += (np.sum(weights[p][:,p]) *
              np.sum(np.triu(covs[p][:,p], k=1)) / (len(p) * (len(p) - 1) / 2))
    for q in subsets:
      if p == q:
        continue
      lac += (np.sum(weights[p][:,q]) *
              np.sum(covs[p][:,q]) / (len(p) * len(q)))
  return lac / (C * V)


def get_perm_dists(expression, tree_path, lut_path=None, device='cuda:0', shuffle_seed=None,
                   gene_group_size=50, subtree_labels=None, decay_factor=10, quoted_node_names=True):

  labels_in_order = expression.index
  if shuffle_seed is not None:
    np.random.seed(shuffle_seed)
    idx = np.arange(len(expression))
    np.random.shuffle(idx)
    expression.index = expression.index[idx]
    expression = expression.loc[labels_in_order]

  if subtree_labels is None:
    subtree_labels = labels_in_order
  
  tree = Tree(tree_path, quoted_node_names=quoted_node_names, format=1)
  _, max_depth = tree.get_farthest_node()
  max_depth = int(max_depth)

  torch.random.manual_seed(1219241)
  tree_dists, _ = get_tree_dists(tree, labels_in_order)
  tree_dists = torch.from_numpy(tree_dists)
  path_weights = torch.triu(torch.exp(- torch.square(tree_dists) / decay_factor), diagonal=1)

  in_sub = np.isin(labels_in_order, subtree_labels)
  if lut_path is not None:
    with open(lut_path, 'rb') as f:
      apn_lut = torch.from_numpy(np.load(f))
    apn_lut = apn_lut[in_sub][:, in_sub][:, :, in_sub].to(device)
    triplets = torch.argwhere(apn_lut)
    print(triplets.shape)
  
  expression = expression.loc[in_sub]
  expression = expression[expression.columns[expression.sum(axis=0) >= 10]]
  path_weights = path_weights[in_sub][:, in_sub]
  tree_dists = tree_dists[in_sub][:, in_sub]

  data = torch.Tensor(expression.values).to(device)
  path_weights = path_weights.to(device)
  tree_dists = tree_dists.to(device)

  data_norm = ((data - data.mean(axis=0))/data.std(axis=0))
  # expression_dists = torch.sqrt(torch.sum((data[None] - data[:, None])**2, axis=2))
  expression_dists = 0
  for left in range(0, expression.shape[1], gene_group_size):
    expression_dists += torch.sum((data[None, :, left:left+gene_group_size] - data[:, None, left:left+gene_group_size])**2, axis=2)
  expression_dists = torch.sqrt(expression_dists)

  depths = []
  seeds = []
  la_xs = []
  dc_xs = []
  if lut_path is not None:
    trip_xs = []
    trip_correct_xs = []
  for depth in trange(1, max_depth+1):
    label_subsets = extract_subsets_with_depth(tree_path, depth, include_labels=subtree_labels)

    perm_seeds = [12345, 66689, 41382, 3838374, 12311, 882321, 121552, 72311, 41217, 91271]
    for perm_seed in perm_seeds:
      perm_ids = get_permutation_ids(expression.index, label_subsets, seed=perm_seed)
      expression_dist_perm = expression_dists[perm_ids, :][:, perm_ids]

      depths.append(depth)
      seeds.append(perm_seed)

      la_xs.append(local_autocorrelation(path_weights, data_norm[perm_ids], gene_group_size=gene_group_size).item())
      dc_xs.append(torch.corrcoef(torch.vstack((tree_dists.flatten(), expression_dist_perm.flatten())))[0, 1].item())

      if lut_path is not None:
        trips = expression_dist_perm[triplets[:,0], triplets[:,1]] - expression_dist_perm[triplets[:,0], triplets[:,2]]
        trip_correct_xs.append((trips < 0).sum().item() / len(trips))
        trip_xs.append(torch.clip(trips + 1, min=0).mean().item()) # margin 1

  df = pd.DataFrame()
  df['depth'] = depths
  df['perm_seed'] = seeds
  df['la'] = la_xs
  df['dc'] = dc_xs
  if lut_path is not None:
    df['t'] = - np.array(trip_xs)
    df['tc'] = trip_correct_xs
    agg = df.groupby(['depth']).agg(
        la_expression_mean=pd.NamedAgg(column="la", aggfunc="mean"),
        la_expression_min=pd.NamedAgg(column="la", aggfunc="min"),
        la_expression_max=pd.NamedAgg(column="la", aggfunc="max"),
        la_expression_std=pd.NamedAgg(column="la", aggfunc="std"),
        dc_expression_mean=pd.NamedAgg(column="dc", aggfunc="mean"),
        dc_expression_min=pd.NamedAgg(column="dc", aggfunc="min"),
        dc_expression_max=pd.NamedAgg(column="dc", aggfunc="max"),
        dc_expression_std=pd.NamedAgg(column="dc", aggfunc="std"),
        tc_expression_mean=pd.NamedAgg(column="tc", aggfunc="mean"),
        tc_expression_min=pd.NamedAgg(column="tc", aggfunc="min"),
        tc_expression_max=pd.NamedAgg(column="tc", aggfunc="max"),
        tc_expression_std=pd.NamedAgg(column="tc", aggfunc="std"),
        t_expression_mean=pd.NamedAgg(column="t", aggfunc="mean"),
        t_expression_min=pd.NamedAgg(column="t", aggfunc="min"),
        t_expression_max=pd.NamedAgg(column="t", aggfunc="max"),
        t_expression_std=pd.NamedAgg(column="t", aggfunc="std"))
  else:
    agg = df.groupby(['depth']).agg(
        la_expression_mean=pd.NamedAgg(column="la", aggfunc="mean"),
        la_expression_min=pd.NamedAgg(column="la", aggfunc="min"),
        la_expression_max=pd.NamedAgg(column="la", aggfunc="max"),
        la_expression_std=pd.NamedAgg(column="la", aggfunc="std"),
        dc_expression_mean=pd.NamedAgg(column="dc", aggfunc="mean"),
        dc_expression_min=pd.NamedAgg(column="dc", aggfunc="min"),
        dc_expression_max=pd.NamedAgg(column="dc", aggfunc="max"),
        dc_expression_std=pd.NamedAgg(column="dc", aggfunc="std"))

  return agg


def get_expected_lac_bmtm_depth_perm(etree, labels_in_order, labels_subset=None, gamma=10):
  tree_dists, mrca_depths = get_tree_dists(etree, labels_in_order)
  _, max_depth = etree.get_farthest_node()
  max_depth = int(max_depth)

  if labels_subset is not None:
    in_sub = np.isin(labels_in_order, labels_subset)
    tree_dists = tree_dists[in_sub][:, in_sub]
    mrca_depths = mrca_depths[in_sub][:, in_sub]
  
  theo_perm_lac = np.zeros(max_depth)
  for j, perm_d in enumerate(range(1, max_depth+1)):
    subsets = extract_subsets_with_depth(etree, depth=perm_d, 
                                         include_labels=labels_in_order if labels_subset is None else labels_subset,
                                         return_ids=True)
    theo_perm_lac[j] = lac_theoretical_perm(mrca_depths, tree_dists, subsets, gamma=gamma)

  df = pd.DataFrame()
  df['depth'] = list(range(1, max_depth+1))
  df['lac'] = theo_perm_lac
  return df


# for single gene (gene dim in expression is replicates)
def lac_empirical(tree_dists, expression, gamma=10):
  weights = np.triu(np.exp(-tree_dists**2/gamma), k=1)
  norm_expression = (expression - np.mean(expression, axis=0, keepdims=True)) / np.std(
      expression, axis=0, keepdims=True, ddof=1)
  N, M = norm_expression.shape
  terms = weights.reshape(N, N, 1) * norm_expression.reshape(1, N, M) * norm_expression.reshape(N, 1, M)
  C = np.sqrt(np.sum(weights**2))
  lacs = np.sum(terms, axis=(0, 1)) / C
  return np.mean(lacs), np.std(lacs, ddof=1)
