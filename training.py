import torch 
from torch import optim
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

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