from torch import nn 
import torch 

from tree_util import get_tree_info


class AutoEncoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, enc_layers, dec_layers=None, non_linearity=nn.GELU()):
    super(AutoEncoder, self).__init__()
    if dec_layers is None:
      dec_layers = enc_layers
    self.enc = nn.Sequential(*([nn.Linear(input_dim, hidden_dim)] + (enc_layers - 1) * [non_linearity, nn.Linear(hidden_dim, hidden_dim)]))
    self.dec = nn.Sequential(*((dec_layers - 1) * [non_linearity, nn.Linear(hidden_dim, hidden_dim)]) + [non_linearity, nn.Linear(hidden_dim, input_dim)])
  
  def forward(self, x, return_z=False):
    z = self.enc(x)
    if return_z:
      return self.dec(z), z
    return self.dec(z)
  

class TreeGeneModel(nn.Module):
  def __init__(self, tree_path, labels_in_order, gene_dim, decay_factor=4, skip_tip_distance=0, learn_gene_weights=False,
               min_weight=0, initial_weight=0.5, quoted_node_names=False, skip_descendents_size=0, temp=1):
    super().__init__()

    self.decay_factor = decay_factor
    self.skip_tip_distance = skip_tip_distance
    self.min_weight = min_weight
    self.initial_weight = initial_weight
    self.temp = temp

    self.etree, self.edges_ends, self.skip_edges_ends, edges, paths, skipped, _ = get_tree_info(tree_path, initial_weight, labels_in_order, 
                                                                  quoted_node_names=quoted_node_names, skip_tip_distance=skip_tip_distance, 
                                                                  skip_descendents_size=skip_descendents_size)
    self.register_buffer('paths_const', torch.from_numpy(paths).float())
    self.register_buffer('skipped_const', torch.from_numpy(skipped).float())
    self.log_edge_weights = nn.Parameter(torch.log(torch.tensor(edges).float()))
    if learn_gene_weights:
      self.log_gene_weights = nn.Parameter(torch.zeros(gene_dim))
    else:
      self.register_buffer('log_gene_weights', torch.zeros(gene_dim))
    
  def forward(self):
    edge_weights = torch.exp(self.log_edge_weights) + self.min_weight
    dist = (self.paths_const * edge_weights[None, None, :]).sum(axis=-1)
    dist += self.skipped_const * self.initial_weight # add weights for constant edges
    path_weights = torch.triu(torch.exp(- torch.square(dist) / self.decay_factor), diagonal=1)
    if self.temp == 1:
      gene_weights = torch.nn.functional.softmax(self.log_gene_weights, dim=0)
    else: 
      gene_weights = torch.exp(self.log_gene_weights / self.temp)
      gene_weights = gene_weights / torch.sum(gene_weights)
    return path_weights, gene_weights
