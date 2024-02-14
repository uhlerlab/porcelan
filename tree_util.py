import numpy as np
from tqdm import trange
import pandas as pd
from Bio import Phylo
from ete3 import Tree


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


def get_tree_dists(etree, labels_in_order, return_paths=False):
  etree = etree.copy()
  leaf_ancestors = {}
  n = 0
  for node in etree.traverse('levelorder'):
    if not node.is_leaf():
      node.add_features(id=n)
      n += 1
    if not node.is_root():
      node.add_features(ancestors=node.up.ancestors + [node.up.id])
    else:
      node.add_features(ancestors=[])
    if node.is_leaf():
      leaf_ancestors[node.name] = node.ancestors

  assert set(leaf_ancestors.keys()) == set(labels_in_order)
  leaves = labels_in_order

  if return_paths:
    paths = np.zeros((len(leaves), len(leaves), n + len(leaves)), dtype=np.uint8)
  mrca_depths = np.zeros((len(leaves), len(leaves)))
  dists = np.zeros((len(leaves), len(leaves)))
  # get indices for edges for path between each pair of leaves P
  for i in range(len(leaves) - 1):
    for j in range(i + 1, len(leaves)):
      a = leaf_ancestors[leaves[i]]
      b = leaf_ancestors[leaves[j]]
      k = 0
      while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
      path = a[k:] + b[k:] + [n+i, n+j] # edge indices, not in order
      if return_paths:
        paths[i, j, path] = 1
        # skip paths[j, i, path] = 1, let's only use upper triangle

      mrca_depths[i, i] = len(a)
      mrca_depths[j, j] = len(b)
      mrca_depths[i, j] = k - 1
      mrca_depths[j, i] = k - 1
      dists[i, j] = len(path)
      dists[j, i] = len(path)
      # print(f'{leaves[i]} -> {leaves[j]} ({len(path)}): ', path)

  if return_paths:
    return dists, mrca_depths, paths
  return dists, mrca_depths


def get_tree_info(tree_path, initial_weight, labels_in_order, quoted_node_names=False, 
                  skip_tip_distance=-1, skip_descendents_size=0, return_mrca_depth=False):
  etree = Tree(tree_path, format=1, quoted_node_names=quoted_node_names)
    # .dist is distance to parent, i.e. length of incoming edge but we ignore it and assume it's 1
  leaf_ancestors = {}
  n, ne, ns = 0, 0, 0
  skip_offset = 3 * len(labels_in_order)
  edges_ends = []
  skip_edges_ends = []
  id_to_name = {}
  for node in etree.traverse('levelorder'):
    if not node.is_leaf():
      if node.name == '':
        node.name = f'N{n:04d}'
      n += 1

      d = node.get_closest_leaf()[1]
      c = len(node.get_leaves())
      # add internal node names
      if d > skip_tip_distance and c > skip_descendents_size:
        node.add_features(id=ne)
        ne += 1
        edges_ends.append(node.name)
      else: 
        node.add_features(id=skip_offset+ns)
        skip_edges_ends.append(node.name)
        ns += 1
      id_to_name[node.id] = node.name
    if not node.is_root():
      node.add_features(ancestors=node.up.ancestors + [node.up.id])
    else:
      node.add_features(ancestors=[])
    if node.is_leaf():
      leaf_ancestors[node.name] = node.ancestors

  assert set(leaf_ancestors.keys()) == set(labels_in_order)
  leaves = labels_in_order
  edges = initial_weight * np.ones(len(edges_ends) + (len(leaves) if skip_tip_distance < 0 else 0))

  paths = np.zeros((len(leaves), len(leaves), len(edges)), dtype=np.uint8)
  skipped = np.zeros((len(leaves), len(leaves)))
  if return_mrca_depth:
    mrca_depths = np.zeros((len(leaves), len(leaves)))
  # get indices for edges for path between each pair of leaves P
  for i in range(len(leaves) - 1):
    for j in range(i + 1, len(leaves)):
      a = leaf_ancestors[leaves[i]]
      b = leaf_ancestors[leaves[j]]
      k = 0
      while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
      path = a[k:] + b[k:] # edge indices, not in order
      if skip_tip_distance < 0 and skip_descendents_size < 1:
        path += [n+i, n+j]
      else:
        path = np.array(path, dtype=int)
        skipped[i, j] = 2 + np.sum(path >= skip_offset)
        path = path[path < skip_offset]
      paths[i, j, path] = 1
      if return_mrca_depth:
        mrca_depths[i, j] = k - 1
      # skip paths[j, i, path] = 1, let's only use upper triangle

  if skip_tip_distance < 0 and skip_descendents_size < 1:
    edges_ends += list(leaves)
  else:
    skip_edges_ends += list(leaves)

  leaf_ancestors_0 = list(map(lambda id: id_to_name[id], leaf_ancestors[leaves[0]]))
  if return_mrca_depth:
    return etree, edges_ends, skip_edges_ends, edges, paths, skipped, mrca_depths, leaf_ancestors_0
  else: 
    return etree, edges_ends, skip_edges_ends, edges, paths, skipped, leaf_ancestors_0
  

def augment_tree_with_weights(tree_path, weights, leaf_weight=None, quoted_node_names=False, root_name=None):
  etree = Tree(tree_path, format=1, quoted_node_names=quoted_node_names)
  n = 0
  for node in etree.traverse('levelorder'):
    if not node.is_leaf():
      # add internal node names
      if node.name == '':
        if node.is_root() and root_name is not None:
          node.name = root_name
        else:
          node.name = f'N{n:04d}'
        n += 1
    if leaf_weight is None or not node.is_leaf():
      node.dist = weights.loc[str(node.name)].item()
    else:
      node.dist = leaf_weight

  return etree

# Adapted from https://chart-studio.plotly.com/~empet/14834.embed
def _get_circular_tree_data(tree, order='level', dist=1, start_angle=0, end_angle=360, start_leaf='first', points_per_arc=(10, 5, 3)):
    """Define  data needed to get the Plotly plot of a circular tree
    """
    # tree:  an instance of Bio.Phylo.Newick.Tree or Bio.Phylo.PhyloXML.Phylogeny
    # order: tree  traversal method to associate polar coordinates to its nodes
    # dist:  the vertical distance between two consecutive leafs in the associated rectangular tree layout
    # start_angle:  angle in degrees representing the angle of the first leaf mapped to a circle
    # end_angle: angle in degrees representing the angle of the last leaf
    # the list of leafs mapped in anticlockwise direction onto circles can be tree.get_terminals() 
    # or its reversed version tree.get_terminals()[::-1]. 
    # start leaf: is a keyword with two possible values"
    # 'first': to map  the leafs in the list tree.get_terminals() onto a circle,
    #         in the counter-clockwise direction
    # 'last': to map  the leafs in the  list, tree.get_terminals()[::-1] 
    
    start_angle *= np.pi/180 # conversion to radians
    end_angle *= np.pi/180
    
    def get_radius(tree):
        """
        Associates to  each clade root its radius, equal to the distance from that clade to the tree root
        returns dict {clade: node_radius}
        """
        node_radius = tree.depths()
        
        #  If the tree did not record  the branch lengths  assign  the unit branch length
        #  (ex: the case of a newick tree "(A, (B, C), (D, E))")
        if not np.count_nonzero(node_radius.values()):
            node_radius = tree.depths(unit_branch_lengths=True)
        return node_radius
   
    
    def get_vertical_position(tree):
        """
        returns a dict {clade: ycoord}, where y-coord is the cartesian y-coordinate 
        of a  clade root in a rectangular phylogram
        
        """
        # Assign y-coordinates to the tree leafs
        if start_leaf == 'first':
            node_ycoord = dict((leaf, k) for k, leaf in enumerate(tree.get_terminals()))
        elif start_leaf == 'last':
            node_ycoord = dict((leaf, k) for k, leaf in enumerate(reversed(tree.get_terminals())))
        else:
            raise ValueError("start leaf can be only 'first' or 'last'")
            
        def assign_ycoord(clade):#compute the y-coord for the root of this clade
            for subclade in clade:
                if subclade not in node_ycoord: # if the subclade root hasn't a y-coord yet
                    assign_ycoord(subclade)
            node_ycoord[clade] = 0.5 * (node_ycoord[clade.clades[0]] + node_ycoord[clade.clades[-1]])

        if tree.root.clades:
            assign_ycoord(tree.root)
        return node_ycoord

    node_radius = get_radius(tree)
    node_ycoord = get_vertical_position(tree)
    y_vals = node_ycoord.values()
    ymin, ymax = min(y_vals), max(y_vals)
    ymin -= dist # this dist subtraction is necessary to avoid coincidence of the  first and last leaf angle
                 # when the interval  [ymin, ymax] is mapped onto [0, 2pi],
                
    def ycoord2theta(y):
        # maps an y in the interval [ymin-dist, ymax] to the interval [radian(start_angle), radian(end_angle)]
        
        return start_angle + (end_angle - start_angle) * (y-ymin) / float(ymax-ymin)


    def get_points_on_lines(linetype='radial', x_left=0, x_right=0, y_right=0,  y_bot=0, y_top=0):
        """
        - define the points that generate a radial branch and the circular arcs, perpendicular to that branch
         
        - a circular arc (angular linetype) is defined by 10 points on the segment of ends
        (x_bot, y_bot), (x_top, y_top) in the rectangular layout,
         mapped by the polar transformation into 10 points that are spline interpolated
        - returns for each linetype the lists X, Y, containing the x-coords, resp y-coords of the
        line representative points
        """
       
        if linetype == 'radial':
            theta = ycoord2theta(y_right) 
            X = [x_left*np.cos(theta), x_right*np.cos(theta), None]
            Y = [x_left*np.sin(theta), x_right*np.sin(theta), None]
        
        elif linetype == 'angular':
            theta_b = ycoord2theta(y_bot)
            theta_t = ycoord2theta(y_top)
            if abs(theta_b - theta_t) > np.pi/6:
              t = np.linspace(0, 1, points_per_arc[0]) 
            elif abs(theta_b - theta_t) > np.pi/10:
              t = np.linspace(0, 1, points_per_arc[1]) 
            else:
              t = np.linspace(0, 1, points_per_arc[2]) # 3 points that span the circular arc 
            theta = (1-t) * theta_b + t * theta_t
            X = list(x_right * np.cos(theta)) + [None]
            Y = list(x_right * np.sin(theta)) + [None]
        
        else:
            raise ValueError("linetype can be only 'radial' or 'angular'")
       
        return X,Y   
        

    def get_line_lists(clade,  x_left,  xlines, ylines, xarc, yarc):
        """Recursively compute the lists of points that span the tree branches"""
        
        # xlines, ylines  - the lists of x-coords, resp y-coords of radial edge ends
        # xarc, yarc - the lists of points generating arc segments for tree branches
        
        x_right = node_radius[clade]
        y_right = node_ycoord[clade]
   
        X,Y = get_points_on_lines(linetype='radial', x_left=x_left, x_right=x_right, y_right=y_right)
   
        xlines.extend(X)
        ylines.extend(Y)
   
        if clade.clades:
           
            y_top = node_ycoord[clade.clades[0]]
            y_bot = node_ycoord[clade.clades[-1]]
       
            X,Y = get_points_on_lines(linetype='angular',  x_right=x_right, y_bot=y_bot, y_top=y_top)
            xarc.extend(X)
            yarc.extend(Y)
       
            # get and append the lists of points representing the  branches of the descedants
            for child in clade:
                get_line_lists(child, x_right, xlines, ylines, xarc, yarc)

    xlines = []
    ylines = []
    xarc = []
    yarc = []
    get_line_lists(tree.root,  0, xlines, ylines, xarc, yarc)  
    xnodes = []
    ynodes = []
    node_labels = []
    terminals = []

    for clade in tree.find_clades(order='preorder'): #it was 'level'
        theta = ycoord2theta(node_ycoord[clade])
        xnodes.append(node_radius[clade]*np.cos(theta))
        ynodes.append(node_radius[clade]*np.sin(theta))
        node_labels.append(clade.name)
        terminals.append(clade.is_terminal())
        
    return node_labels, terminals, xnodes, ynodes, xlines, ylines, xarc, yarc    


def get_node_and_edge_data(tree_path, include_labels=None, points_per_arc=(10, 5, 3), end_angle=360):
  ptree = Phylo.read(tree_path, 'newick')
  node_labels, terminals, xnodes, ynodes, xlines, ylines, xarc, yarc = _get_circular_tree_data(
          ptree, start_leaf='last', points_per_arc=points_per_arc, end_angle=end_angle)

  if include_labels is not None:
    node_labels = np.array(node_labels)
    node_labels[~np.isin(node_labels, include_labels)] = None

  def separate_sources_targets(coords):
    s = []
    t = []
    i = 0
    while i < len(coords) - 1:
      while coords[i] is not None and coords[i+1] is not None and i < len(coords) - 1:
        s.append(coords[i])
        t.append(coords[i+1])
        i += 1
      i += 1 # skip None
    return s, t

  lines_sx, lines_tx = separate_sources_targets(xlines)
  lines_sy, lines_ty = separate_sources_targets(ylines)
  arcs_sx, arcs_tx = separate_sources_targets(xarc)
  arcs_sy, arcs_ty = separate_sources_targets(yarc)
  node_data = pd.DataFrame.from_records(zip(node_labels, terminals, xnodes, ynodes), columns=['cell_label', 'is_leaf', 'x', 'y'])
  edge_data = pd.DataFrame.from_records(zip(lines_sx + arcs_sx, lines_sy + arcs_sy, lines_tx + arcs_tx, lines_ty + arcs_ty), columns=['sx', 'sy', 'tx', 'ty'])
  
  return node_data, edge_data