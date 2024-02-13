import numpy as np
from tqdm import trange


def _cell_dist(pdm, c1, c2):
  return pdm(pdm.taxon_namespace.get_taxon(c1), pdm.taxon_namespace.get_taxon(c2))


def dist_matrix_to_numpy(pdm, names):
  n = len(names)
  dists = np.zeros((n, n))
  for i in range(n - 1):
    for j in range(i, len(names)):
      dists[i][j] = _cell_dist(pdm, names[i], names[j])
      dists[j][i] = dists[i][j]
  return dists


def get_apn_dist_triplet_lut(dist_matrix):
  n = len(dist_matrix)
  apn_lut = np.zeros((n, n, n), dtype=bool)
  for a in trange(len(dist_matrix)):
    pn_order_neg_minus_pos = - dist_matrix[a].reshape(-1, 1) + dist_matrix[a].reshape(1, -1)
    # neg - pos >= ||a - pos||
    apn_lut[a] = pn_order_neg_minus_pos >= dist_matrix[a].reshape(-1, 1)
    apn_lut[a, a, :] = 0
    apn_lut[a, :, a] = 0
  return apn_lut
