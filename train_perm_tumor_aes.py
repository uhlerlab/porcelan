import numpy as np
import pandas as pd
from training import train_triplet
from perm_util import extract_subsets_with_depth, get_permutation_ids, split_by_type
from tqdm import trange
import argparse


def main(training_seed, max_depth, device, model_suffix, num_epochs=10, save_epochs=[], h=1, fixed_types=True):
  print('Training seed', training_seed, 'Device', device)

  pre_train_path = MODEL_PREFIX + MODEL_PRETRAIN_SUFFIX
  with open(CELLS_PATH) as f:
    labels_in_order = np.array(f.read().splitlines()) 
  with open(TYPES_PATH) as f:
    cell_types_in_order = np.array(f.read().splitlines()) 
  cell_types = pd.DataFrame(list(zip(labels_in_order, cell_types_in_order)), columns=['name', 'type']).set_index('name')
  expression = pd.read_csv(EXPRESSION_PATH)
  with open(GENES_PATH) as f:
    genes = np.array(f.read().splitlines()) 
  expression = expression[genes].values
  dim = len(genes)

  for depth in trange(0, max_depth+1):
    if depth > 0:
      subtrees = extract_subsets_with_depth(TREE_PATH, depth=depth, include_labels=labels_in_order)
      label_subsets = []
      for stree in subtrees:
        if fixed_types:
          label_subsets.extend(split_by_type(stree, cell_types))
        else:
          label_subsets.append(stree)

    perm_seeds = [12345, 66689, 41382, 3838374, 12311] #, 882321, 121552, 72311, 41217, 91271]
    for perm_seed in perm_seeds:
      if depth > 0:
        perm_ids = get_permutation_ids(labels_in_order, label_subsets, seed=perm_seed)
        assert (not fixed_types) or (cell_types_in_order[perm_ids] == cell_types_in_order).all()
      elif perm_seed != perm_seeds[0]:
        continue  # nothing to permute for depth 0, so skip all but first seed 
      else:
        perm_ids = None
      _ = train_triplet(model=None, 
              model_path=MODEL_PREFIX + MODEL_KIND + '_lr1em4_e{:d}_b128_h{:s}_pd_pre_ts{:d}_perm{:d}d{:d}{:s}_g{:d}{:s}.pt'.format(
                      num_epochs, f'{h:0.2f}'.replace('.', 'p') if np.ceil(h) != np.floor(h) else str(h), training_seed, perm_seed, depth, 'type' if fixed_types else '', dim, model_suffix), 
                      training_seed=training_seed, n_genes=dim, h=h, gene_matrix=expression, apn_lut_path=LUT_PATH, 
                      display=False, num_epochs=num_epochs, device=device, batch_size=128, lr=1e-4, init_path=pre_train_path, 
                      perm_data_ids=perm_ids, save_epochs=save_epochs)
      
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--tumor", required=True, type=str)
  parser.add_argument("-m", "--model_suffix", required=False, type=str, default='')
  parser.add_argument("-l", "--lmbda", required=False, type=float, default=1)
  parser.add_argument("-d", "--max_depth", required=True, type=int)
  parser.add_argument("-c", "--device", required=False, type=str, default='cuda:0')
  parser.add_argument("-e", "--epochs", required=False, type=int, default=100)
  parser.add_argument("-s", "--save_epochs", required=False, type=int, nargs='*')
  parser.add_argument("-k", "--model_kind", required=False, type=str, default='AELR-2-1000')
  parser.add_argument("-a", "--permute_all_types", required=False, action='store_true')
  args = parser.parse_args()

  name = args.tumor
  if name.startswith('tedsim'):
    TREE_PATH = f'data/tedsim/{name}_pruned.nwk'
    CELLS_PATH = f'data/tedsim/{name}_cells.txt'
    TYPES_PATH = f'data/tedsim/{name}_cell_types.txt'
    EXPRESSION_PATH = f'data/tedsim/{name}_normalized_log_counts.txt'
    GENES_PATH = f'data/tedsim/{name}{args.model_suffix}_genes.txt'
    LUT_PATH = f'data/tedsim/{name}_apn_pd_triplet_lut.npy'
  else:
    TREE_PATH = f'data/preprocessed/{name}_pruned.nwk'
    CELLS_PATH = f'data/preprocessed/{name}_cells.txt'
    TYPES_PATH = f'data/preprocessed/{name}_cell_types.txt'
    EXPRESSION_PATH = f'data/preprocessed/{name}_normalized_log_counts.txt'
    GENES_PATH = f'data/preprocessed/{name}{args.model_suffix}_genes.txt'
    LUT_PATH = f'data/preprocessed/{name}_apn_pd_triplet_lut.npy'
  MODEL_PREFIX = f'results/{name}_'
  MODEL_PRETRAIN_SUFFIX = f'{args.model_kind}_lr1em4_e500_b128{args.model_suffix}.pt'
  MODEL_KIND = args.model_kind

  for training_seed in [112221, 6243321, 99483, 92231, 555242]:
    main(training_seed, args.max_depth, device=args.device, model_suffix=args.model_suffix, fixed_types=not args.permute_all_types,
         num_epochs=args.epochs, save_epochs=args.save_epochs if args.save_epochs is not None else [], h=args.lmbda)

# example usage:
# python train_perm_tumor_aes.py -t '3435_NT_T1' -m '_hvg' -d 17 -c 'cuda:2' -k 'AELR-3-1000'
