import torch 
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scores import local_autocorrelation


# expression should be a normalized vector
# gene_group_size does not affect result but setting this will save memory by computing the result in multiple steps of the specified size 
def optimize_weights(model, expression, lr=1, epochs=100, device='cuda:0', noise_seed=12345, gene_group_size=None, use_float=False):
  torch.manual_seed(noise_seed)

  expression = torch.tensor(expression)
  if use_float:
    expression = expression.float()
  expression = expression.to(device)
  model = model.to(device).train()
  optimizer = optim.Adam(model.parameters(), lr=lr)

  scores = []
  tqdm_range = trange(epochs)
  for e in tqdm_range:
    path_weights, gene_weights = model.forward()

    loss = -local_autocorrelation(path_weights, expression, lambda x: (gene_weights * x).sum(), gene_group_size=gene_group_size)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    scores.append(-loss.item())
    tqdm_range.set_description(f"epoch: {e + 1}; score: {scores[-1]:.15f}; ")

  fig, ax = plt.subplots(figsize=(5,3))
  ax.set_yscale('log')
  ax.set_ylabel('log score')
  ax.set_xlabel('iterations', color='k')
  ax.plot(np.arange(len(scores)), scores, label='Local autocorrelation')
  fig.tight_layout()

  return fig


# -------------------------  autoencoder training -------------------------------------

training_seed = 1234121
def set_seeds(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)


def train(model, model_path, expression_path=None, gene_matrix=None, num_epochs=1000, lr=1e-3, device="cuda:1", 
          scheduler_factory=None, train_val_split=None, batch_size=None, n_genes=100, training_seed=training_seed):

  if expression_path is not None:
    data = torch.Tensor(pd.read_csv(expression_path, sep=' ', header=None).values[:, :n_genes])
  else:
    assert gene_matrix is not None
    data = torch.Tensor(gene_matrix[:, :n_genes])
  if train_val_split is not None:
    train_data = data[train_val_split[0]]
    val_data = data[train_val_split[1]]
  else:
    train_data = data

  train_losses = []
  val_losses = []
  set_seeds(training_seed)
      
  tqdm_range = tqdm(np.arange(num_epochs)) 
  model = model.to(device)
  train_data = train_data.to(device)
  if train_val_split is not None:
    val_data = val_data.to(device)
  model.train()   
  optimizer = optim.Adam(model.parameters(), lr=lr)  
  if scheduler_factory is not None: 
    scheduler = scheduler_factory(optimizer)

  if batch_size is not None: 
    train_dataloader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
  for e in tqdm_range:
    if batch_size is None:
      x_rec, x_z = model.forward(train_data, return_z=True)
      rec_loss = F.mse_loss(x_rec, train_data)

      optimizer.zero_grad()
      rec_loss.backward()
      optimizer.step()

      train_losses.append(rec_loss.item())
    else: 
      epoch_loss = 0
      batch_count = 0
      for batch_data, in train_dataloader:
        x_rec, _ = model.forward(batch_data, return_z=True)
        rec_loss = F.mse_loss(x_rec, batch_data)

        optimizer.zero_grad()
        rec_loss.backward()
        optimizer.step()
        epoch_loss += rec_loss.item()
        batch_count += 1
      
      train_losses.append(epoch_loss / batch_count)

    if scheduler_factory is not None:
      scheduler.step()

    if train_val_split is not None:
      model.eval()
      with torch.no_grad():
        x_rec, x_z = model.forward(val_data, return_z=True)
        rec_loss = F.mse_loss(x_rec, val_data)
      model.train()
      val_losses.append(rec_loss.item())

    if train_val_split is not None:
      tqdm_range.set_description(
          (
              f"epoch: {e + 1}; train recon: {train_losses[-1]:.15f}; val recon: {val_losses[-1]:.15f}"
          )
      )
    else:
      tqdm_range.set_description(
          (
              f"epoch: {e + 1}; recon: {train_losses[-1]:.15f}; "
          )
      )

  torch.save(model, model_path)
  df = pd.DataFrame()
  df['epoch'] = np.arange(num_epochs)
  df['train_loss'] = train_losses
  if train_val_split is not None:
    df['val_loss'] = val_losses
  df.to_csv(model_path[:-3] + '_loss.csv', index=False)

  if train_val_split is not None:
    fig = _get_loss_plot(None, train_losses, val_losses, loss_names=['', 'train recon', 'val recon'])
  else:
    fig = _get_loss_plot(None, train_losses, None)
  return fig


def _get_loss_plot(losses1, losses2, losses3, loss_names=['Triplet loss', 'Recon. loss', 'Total loss'], xlabel='epochs'):
  fig, ax = plt.subplots(figsize=(10,7))
  ax.set_yscale('log')
  ax.set_ylabel('log loss', fontsize=20)
  ax.set_xlabel(xlabel, color='k', fontsize=20)
  if losses1 is not None:
    ax.plot(np.arange(len(losses1)), losses1, label=loss_names[0])
  if losses2 is not None:
    ax.plot(np.arange(len(losses2)), losses2, label=loss_names[1])
  if losses3 is not None:
    ax.plot(np.arange(len(losses3)), losses3, label=loss_names[2])
  ax.legend(prop={'size': 20})
  ax.tick_params(axis='both', which='major', labelsize=15)
  fig.tight_layout()
  return fig


def trip_loss_func(X, apn_triplets, dist_func=None):
  # default dist_func is 2-norm of vector difference, i.e., Euclidean distance
  if torch.is_tensor(apn_triplets):
    apn_triplets = apn_triplets.long()
  return F.triplet_margin_with_distance_loss(
      anchor=X[apn_triplets[:, 0]], 
      positive=X[apn_triplets[:, 1]], 
      negative=X[apn_triplets[:, 2]], 
      distance_function=dist_func)


def _eval_all(model, data, apn_lut):
  print('Evaluating model ...')

  with torch.no_grad():
    x_rec, x_z = model.forward(data, return_z=True)
    rec_loss = F.mse_loss(x_rec, data)

  print('Total rec loss:', rec_loss.item())

  if np.product(apn_lut.shape) < 2147483647:  # INT_MAX
    triplets = torch.argwhere(apn_lut)
  else: 
    triplets = []
    for a in range(len(apn_lut)):
      triplets.extend([(a, p, n) for (p, n) in torch.argwhere(apn_lut[a])])
    triplets = torch.tensor(triplets)
  trip_loss = 0
  trip_batch_size = 3000
  for i in range(0, len(triplets), trip_batch_size):
    trips = triplets[i:i+trip_batch_size]
    with torch.no_grad():
      trip_loss += len(trips) * trip_loss_func(x_z, trips) / len(triplets)

  print('Total trip loss:', trip_loss.item())


def train_triplet(model, model_path, expression_path=None, apn_lut_path=None, gene_matrix=None, batch_size=128,
          num_epochs=1000, lr=1e-3, h=1, device="cuda:1", dist_func=None,
          perm_data_seed=None, perm_data_ids=None, resample_data_seed=None, n_genes=100, init_path=None,
          training_seed=training_seed, display=True, lut_on_device=True, save_epochs='last', 
          apn_subset_indexer=None, save_final_checkpoint=False, eval_final_all_if_display=True):

  if init_path is not None:
    print('loading model from path', init_path)
    model = torch.load(init_path, map_location=device)
    print('done loading model from path')

  with open(apn_lut_path, 'rb') as f:
    apn_lut = torch.from_numpy(np.load(f))
  if apn_subset_indexer is not None:
    apn_lut = apn_lut[apn_subset_indexer][:, apn_subset_indexer][:, :, apn_subset_indexer]
    

  if expression_path is not None:
    data = torch.Tensor(pd.read_csv(expression_path, sep=' ', header=None).values[:, :n_genes])
  else:
    assert gene_matrix is not None
    data = torch.Tensor(gene_matrix[:, :n_genes])

  if perm_data_seed is not None:
    torch.manual_seed(perm_data_seed)
    data = data[torch.randperm(len(data))]
  elif perm_data_ids is not None:
    data = data[perm_data_ids]
  elif resample_data_seed is not None:
    np.random.seed(resample_data_seed)
    data = data[np.random.choice(len(data), size=len(apn_lut))]
  
  if lut_on_device:
    apn_lut = apn_lut.to(device)
  data = data.to(device)
  model = model.to(device)
  model.eval()
  with torch.no_grad():
    x_rec = model.forward(data, return_z=False)
    rec_loss = F.mse_loss(x_rec, data).item()
  print('Pre-trained recon. loss is', rec_loss)

  train_idx_loader = DataLoader(TensorDataset(torch.arange(len(data))), batch_size=batch_size, shuffle=True)
  
  model.train()   
  optimizer = optim.Adam(model.parameters(), lr=lr)

  trip_losses = []
  rec_losses = []
  total_losses = []
  set_seeds(training_seed)
      
  assert next(model.parameters()).is_cuda
  assert data.is_cuda

  tqdm_range = tqdm(np.arange(num_epochs)) 
  for e in tqdm_range:
    for batch_idx, in train_idx_loader:
      batch_data = data[batch_idx]
      x_rec, x_z = model.forward(batch_data, return_z=True)
      rec_loss = F.mse_loss(x_rec, batch_data)
      batch_lut = apn_lut[batch_idx][:, batch_idx][:, :, batch_idx]
      if not lut_on_device:
        batch_lut = batch_lut.to(device)
      triplets = torch.argwhere(batch_lut)
      trip_loss = trip_loss_func(x_z, triplets, dist_func=dist_func)

      loss = rec_loss + h * trip_loss
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
      rec_losses.append(rec_loss.item())
      trip_losses.append(trip_loss.item())
      total_losses.append(loss.item())

    tqdm_range.set_description(
        (
            f"epoch: {e + 1}; recon: {rec_losses[-1]:.15f}; "
            f"trip: {trip_losses[-1]:.13f}; total: {total_losses[-1]:.15f}; "
        )
    )

    if save_epochs != 'last' and (e + 1) in save_epochs:
      torch.save(model, model_path[:-3] + f'_e{e + 1}.pt')
      print('Saved model at', model_path[:-3] + f'_e{e + 1}.pt')

  if save_epochs == 'last':
    torch.save(model, model_path)
    print('Saved model at', model_path)

  if save_final_checkpoint:
    checkpoint = { 
          'epoch': e+1,
          'model': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'recon_loss': rec_losses, 
          'trip_loss': trip_losses, 
          'total_loss': total_losses}
    torch.save(checkpoint, model_path[:-3] + '_checkpoint.pt')
    print('Saved checkpoint at', model_path[:-3] + '_checkpoint.pt')

  df = pd.DataFrame()
  df['iter'] = np.arange(num_epochs * len(train_idx_loader))
  df['recon_loss'] = rec_losses
  df['trip_loss'] = trip_losses
  df['total_loss'] = total_losses
  df.to_csv(model_path[:-3] + '_loss.csv', index=False)

  if display:
    if eval_final_all_if_display:
        _eval_all(model, data, apn_lut)
    fig = _get_loss_plot(trip_losses, rec_losses, total_losses, xlabel='iteration')
    return fig
  return None
