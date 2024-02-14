import pandas as pd
import numpy as np
import altair as alt
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


kp_color_domain = list(range(29)) + [
  None,
 'AT1-like',
 'AT2-like',
 'Apc Early',
 'Apc Mesenchymal-1 (Met)',
 'Apc Mesenchymal-2',
 'Early EMT-1',
 'Early EMT-2',
 'Early gastric',
 'Endoderm-like',
 'Gastric-like',
 'High plasticity',
 'Late Gastric',
 'Lkb1 subtype',
 'Lung progenitor-like',
 'Mesenchymal-1',
 'Mesenchymal-1 (Met)',
 'Mesenchymal-2',
 'Mesenchymal-2 (Met)',
 'Pre-EMT']


nice_colors = ['#bcbcbc',
               '#5d00c3',
               '#1f77b4',
               '#ff7f0e',
               '#00DCDC',
               '#2ca02c',
               '#FED000',
               '#d62728',
               '#9467bd',
               '#8c564b',
               '#e377c2',
               '#666666',
               '#bcbd22',
               '#17becf',
               '#66c2a5',
               '#fc8d62',
               '#8da0cb',
               '#e78ac3',
               '#a6d854',
               '#ffd92f',
               '#e5c494',
               '#232323',
               '#fbb4ae',
               '#b3cde3',
               '#ccebc5',
               '#decbe4',
               '#fed9a6',
               '#ffffcc',
               '#e5d8bd',
               # repeat 
               '#bcbcbc',
               '#5d00c3',
               '#1f77b4',
               '#ff7f0e',
               '#00DCDC',
               '#2ca02c',
               '#FED000',
               '#d62728',
               '#9467bd',
               '#8c564b',
               '#e377c2',
               '#666666',
               '#bcbd22',
               '#17becf',
               '#66c2a5',
               '#fc8d62',
               '#8da0cb',
               '#f8befa',
               '#a6d854',
               '#bfe6ff']


def _augment_source_with_labels(node_data, cell_labels, clusters, cluster_names):
  node_data = node_data.set_index('cell_label')
  cell_info = pd.DataFrame.from_records(zip(cell_labels, *clusters), columns=['cell_label'] + cluster_names).set_index('cell_label')
  return node_data.join(cell_info).reset_index()

def _augment_source_with_leaf_embedding(node_data, cell_labels, embed_xy, suffix):
  node_data = node_data.set_index('cell_label')
  embed_df = pd.DataFrame.from_records(zip(cell_labels, embed_xy[:,0], embed_xy[:,1]), columns=['cell_label', 'embed_x' + suffix, 'embed_y' + suffix]).set_index('cell_label')
  return node_data.join(embed_df).reset_index()


def single_tree_plot(nodes, edges, cell_labels, colors, color_names,
                     ordinal_colors=False, continuous_colors=False,  use_kp_domain=True,
                     disable_grid=True, show_legend=True, tooltip_names=None,
                     width=350, node_size=30, additional_colors=[], additional_color_domain=[]):
  
  nodes = _augment_source_with_labels(nodes, cell_labels, colors, color_names)
  cluster_dropdown = alt.selection_single(fields=['color'],
                                     bind=alt.binding_select(options=color_names, name='color'),
                                     init={'color': color_names[0]})

  if ordinal_colors:
    cond_color = alt.Color('color_value:O', scale=alt.Scale(scheme='redyellowgreen'), legend=alt.Legend() if show_legend else None) 
  elif continuous_colors:
    cond_color = alt.Color('color_value:Q', scale=alt.Scale(scheme='viridis'), legend=alt.Legend() if show_legend else None)
  else:
    color_range = list(nice_colors)
    if use_kp_domain:
      cond_color = alt.Color('color_value:N', 
                             scale=alt.Scale(range=color_range + additional_colors, domain=kp_color_domain + additional_color_domain), 
                             legend=alt.Legend(columns=8, symbolLimit=0) if show_legend else None)
    else: 
      cond_color = alt.Color('color_value:N', scale=alt.Scale(range=color_range + additional_colors), 
                             legend=alt.Legend(columns=8, symbolLimit=0) if show_legend else None)


  if tooltip_names is None:
    ['cell_label'] + color_names
  nodes = alt.Chart(nodes).mark_circle().transform_fold(
    fold=color_names,
    as_=['color', 'color_value']
  ).transform_filter(
    cluster_dropdown
  ).encode(
      color=alt.condition('datum.color_value !== 0.0012345678', cond_color, alt.value('gray')),
      # color=alt.condition('isValid(datum.color_value)', cond_color, alt.value('gray')),
      tooltip=tooltip_names,
      size=alt.SizeValue(node_size)
  ).properties(
    width=width,
    height=width
  ).add_selection(cluster_dropdown)

  links = alt.Chart(edges).mark_rule(
      stroke="#ccc"
  )

  tree = links.encode(x='sx:Q', x2='tx:Q', y='sy:Q', y2='ty:Q') + nodes.encode(x='x:Q', y='y:Q').interactive()

  if disable_grid:
    return tree.configure_axis(grid=False, disable=True).configure_view(strokeWidth=0)
  else: 
    return tree


def _generate_circle(r, n_lines=100):
  xs = r * np.sin(2*np.pi/n_lines * np.arange(n_lines))
  ys = r * np.cos(2*np.pi/n_lines * np.arange(n_lines))
  df = pd.DataFrame.from_records(zip(xs, ys, np.roll(xs, -1),  np.roll(ys, -1)), columns=['sx', 'sy', 'tx', 'ty'])
  return df


def interactive_plot(nodes, edges, cell_labels, clusters, cluster_names, 
                     embed_xy_list, embed_xy_title_list, title_tree='lineage tree',
                     ordinal_colors=False, continuous_colors=False, add_unit_circle=None, plots_per_row=4, 
                     disable_grid=True, show_internal_labeled=False, use_kp_domain=True, additional_colors=[]):
  nodes = _augment_source_with_labels(nodes, cell_labels, clusters, cluster_names)
  for i, embed_xy in enumerate(embed_xy_list):
    nodes = _augment_source_with_leaf_embedding(nodes, cell_labels, embed_xy, '_' + str(i))

  brush = alt.selection(type='interval', resolve='global')
  cluster_dropdown = alt.selection_single(fields=['cluster'],
                                     bind=alt.binding_select(options=cluster_names, name='cluster'),
                                     init={'cluster': cluster_names[0]})
  
  if ordinal_colors:
    cond_color = alt.Color('color:O', scale=alt.Scale(scheme='redyellowgreen')) 
  elif continuous_colors:
    cond_color = alt.Color('color:Q', scale=alt.Scale(scheme='viridis'))
  else:
    color_range = list(nice_colors)
    # if skip_pink:
    #   color_range.remove('#e377c2')
    # if not nodes[nodes['is_leaf']].eq(0).any().any() and not nodes[nodes['is_leaf']].isna().any().any():
    #   color_range = color_range[1:]  # first is reserved for N/A
    if use_kp_domain:
      cond_color = alt.Color('color:N', scale=alt.Scale(range=color_range + additional_colors, domain=kp_color_domain + additional_colors), legend=alt.Legend(columns=8, symbolLimit=0))
    else: 
      cond_color = alt.Color('color:N', scale=alt.Scale(range=color_range + additional_colors), legend=alt.Legend(columns=8, symbolLimit=0))

  if show_internal_labeled:
    important = nodes['cell_label'] != None
  else:
    important = nodes['is_leaf']
  base = alt.Chart(nodes[important]).mark_circle().transform_fold(
    fold=cluster_names,
    as_=['cluster', 'color']
  ).transform_filter(
    cluster_dropdown
  ).encode(
      color=alt.condition(
          brush, 
          cond_color, 
          alt.ColorValue('gray')),
      tooltip=['cell_label'] + cluster_names,
      size=alt.condition(brush, alt.SizeValue(30), alt.SizeValue(10)),
  ).properties(
    width=350,
    height=350
  ).add_selection(cluster_dropdown)

  internal_nodes = alt.Chart(nodes[~important]).mark_circle(size=10).encode(
    x="x:Q",
    y="y:Q",
    color=alt.ColorValue('gray')
  )

  links = alt.Chart(edges).mark_rule(
      stroke="#ccc"
  ).encode(
    x='sx:Q', 
    x2='tx:Q',
    y='sy:Q',
    y2='ty:Q'
  )

  tree = internal_nodes + links + base.encode(x='x:Q', y='y:Q')
  tree = tree.properties(title = alt.TitleParams(text=title_tree, fontSize=15)).interactive()

  if add_unit_circle is not None:
    circle_lines = _generate_circle(1)
    circle = alt.Chart(circle_lines).mark_rule(stroke="#ccc").encode(
            x='sx:Q', x2='tx:Q', y='sy:Q', y2='ty:Q')

  scatters = []
  for i in range(len(embed_xy_list)):
    scatter = base.encode(
            x='embed_x_' + str(i) + ':Q', y='embed_y_' + str(i) + ':Q'
        ).add_selection(
            brush
        ).properties(
            title = alt.TitleParams(text=embed_xy_title_list[i], fontSize=15)
        )
    if add_unit_circle is not None and add_unit_circle[i]:
      scatter = circle + scatter
    scatters.append(scatter)
  plot = (scatters[0] | tree | scatters[1])
  for i in range(2, len(embed_xy_list), plots_per_row):
    row = scatters[i] 
    for j in range(i + 1, min(i + plots_per_row, len(embed_xy_list))):
      row |= scatters[j]
    plot = alt.vconcat(plot, row, center=True)
  if disable_grid:
    return plot.configure_axis(grid=False, disable=True)
  else: 
    return plot


NAN_COLOR_VALUE = 0.0012345678  # altair isValid doesn't seem to catch np.nan -- hopefully this value is not likely to appear in real data on accident
# nodes should have x_a, y_a and x_b, y_b
# edges should have sx_a, sy_a, tx_a, ty_a and tx_b, ty_b, tx_b, ty_b
def double_tree_plot(nodes, edges, cell_labels, colors, color_names,
                     ordinal_colors=False, continuous_colors=False,  use_kp_domain=True,
                     disable_grid=True, show_legend=True, tooltip_names=None,
                     width=350, node_size=30, additional_colors=[], additional_color_domain=[]):
  
  nodes = _augment_source_with_labels(nodes, cell_labels, colors, color_names)
  cluster_dropdown = alt.selection_single(fields=['color'],
                                     bind=alt.binding_select(options=color_names, name='color'),
                                     init={'color': color_names[0]})
  single = alt.selection_single(on='mouseover', nearest=True, resolve='global')
  
  if ordinal_colors:
    cond_color = alt.Color('color_value:O', scale=alt.Scale(scheme='redyellowgreen'), legend=alt.Legend() if show_legend else None) 
  elif continuous_colors:
    cond_color = alt.Color('color_value:Q', scale=alt.Scale(scheme='viridis'), legend=alt.Legend() if show_legend else None)
  else:
    color_range = list(nice_colors)
    if use_kp_domain:
      cond_color = alt.Color('color_value:N', scale=alt.Scale(range=color_range + additional_colors, domain=kp_color_domain + additional_color_domain), legend=alt.Legend(columns=8, symbolLimit=0) if show_legend else None)
    else: 
      cond_color = alt.Color('color_value:N', scale=alt.Scale(range=color_range + additional_colors), legend=alt.Legend(columns=8, symbolLimit=0) if show_legend else None)


  if tooltip_names is None:
    ['cell_label'] + color_names
  nodes = alt.Chart(nodes).mark_circle().transform_fold(
    fold=color_names,
    as_=['color', 'color_value']
  ).transform_filter(
    cluster_dropdown
  ).encode(
      color=alt.condition('datum.color_value !== 0.0012345678', cond_color, alt.value('gray')),
      tooltip=tooltip_names,
      size=alt.condition(~single, alt.SizeValue(node_size), alt.SizeValue(10*node_size))
  ).properties(
    width=width,
    height=width
  ).add_selection(cluster_dropdown, single)

  links = alt.Chart(edges).mark_rule(
      stroke="#ccc"
  )

  tree_left = links.encode(x='sx_a:Q', x2='tx_a:Q', y='sy_a:Q', y2='ty_a:Q') + nodes.encode(x='x_a:Q', y='y_a:Q')
  tree_right = links.encode(x='sx_b:Q', x2='tx_b:Q', y='sy_b:Q', y2='ty_b:Q') + nodes.encode(x='x_b:Q', y='y_b:Q')
  tree = tree_left.interactive() | tree_right.interactive()

  if disable_grid:
    return tree.configure_axis(grid=False, disable=True).configure_view(strokeWidth=0)
  else: 
    return tree



def get_tsne(features, perplexity=30):
  return TSNE(n_components=2, perplexity=perplexity).fit_transform(features)

def get_umap(features):
  return umap.UMAP().fit_transform(features)

def get_pca2(features):
  return PCA(n_components=2).fit_transform(features)
