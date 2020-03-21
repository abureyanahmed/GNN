import sys
import networkx as nx
import random
from subprocess import call
import copy

def greedy_spanner(G, r):
  G_S = nx.Graph()
  for a, b, data in sorted(G.edges(data=True), key=lambda x: x[2]['weight']):
    #print('{a} {b} {w}'.format(a=a, b=b, w=data['weight']))
    if not (a in G_S.nodes() and b in G_S.nodes() and nx.has_path(G_S, a, b)):
      G_S.add_weighted_edges_from([(a, b, G.get_edge_data(a, b)['weight'])])
    else:
      sp = nx.shortest_path_length(G_S, a, b, 'weight')
      if r*data['weight'] < sp:
        G_S.add_weighted_edges_from([(a, b, G.get_edge_data(a, b)['weight'])])
  return G_S

def verify_spanner_with_checker(G_S, G, all_pairs, check_stretch, param):
        for i in range(len(all_pairs)):
                if not (all_pairs[i][0] in G_S.nodes() and all_pairs[i][1] in G_S.nodes()):
                        return False
                if not nx.has_path(G_S, all_pairs[i][0], all_pairs[i][1]):
                        return False
                sp = nx.shortest_path_length(G, all_pairs[i][0], all_pairs[i][1], 'weight')
                #if not check_stretch(nx.dijkstra_path_length(G_S, all_pairs[i][0], all_pairs[i][1]), sp, param):
                if not check_stretch(nx.shortest_path_length(G_S, all_pairs[i][0], all_pairs[i][1], 'weight'), sp, param):
                        return False
        return True

def all_pairs_from_subset(s):
        all_pairs = []
        for i in range(len(s)):
                for j in range(i+1, len(s)):
                        p = []
                        p.append(s[i])
                        p.append(s[j])
                        all_pairs.append(p)
        return all_pairs

def multiplicative_check(subgraph_distance, actual_distance, multiplicative_stretch):
        if subgraph_distance <= multiplicative_stretch*actual_distance:
                return True
        return False

def prune(G, G_main, subset, checker, param):
  G_S = nx.Graph()
  G_S.add_weighted_edges_from(G.edges(data='weight'))
  for e in G.edges():
    G_S.remove_edge(e[0], e[1])
    if not verify_spanner_with_checker(G_S, G_main, all_pairs_from_subset(subset), checker, param):
      G_S.add_edge(e[0], e[1], weight=G.get_edge_data(e[0], e[1])['weight'])
    #else:
    #  print('Pruned:', e[0], e[1])
  return G_S

def top_down(G, subset_arr, checker, param):
  MLG_S = []
  l = len(subset_arr)
  # Traverse the layers top to down
  for i in range(l):
    # copy all the edges from upper level
    G_S = nx.Graph()
    if i>0:
      G_S.add_weighted_edges_from(MLG_S[i-1].edges(data='weight'))
    # cumpute the abstract graph based on the subset
    G_abs = nx.Graph()
    all_pairs = all_pairs_from_subset(subset_arr[i])
    edges_abs=[]
    for p in all_pairs:
      edges_abs.append((p[0], p[1], nx.shortest_path_length(G, p[0], p[1], 'weight')))
    G_abs.add_weighted_edges_from(edges_abs)
    # compute greedy spanner on this graph
    G_abs = greedy_spanner(G_abs, param)
    # add the edges from the greedy spanner
    for u, v in G_abs.edges():
      pth = nx.dijkstra_path(G, u, v)
      for j in range(1, len(pth)):
        G_S.add_edge(pth[j-1], pth[j], weight=G[pth[j-1]][pth[j]]['weight'])
    G_S = prune(G_S, G, subset_arr[i], checker, param)
    MLG_S.append(G_S)
  return MLG_S

data_size = 100
#data_size = 1

root_folder = "experiment_3"
call(["mkdir", root_folder])
name_of_graph_class = ['ER']
number_of_nodes_progression = [10]
number_of_levels = [1]
node_distribution_in_levels = ['L']
cl, l, nd = 0, 0, 0
common_part_of_name = name_of_graph_class[cl]+'_'+str(number_of_nodes_progression[l])+'_'+str(number_of_levels[l])+'_'+node_distribution_in_levels[nd]

def generate_graphs():
  global root_folder, name_of_graph_class, number_of_nodes_progression, number_of_levels, node_distribution_in_levels, common_part_of_name
  for i in range(data_size):
    n = 10
    # for now we have kept the graphs pretty dense, because the graphs are small
    param1 = .5
    G = nx.generators.random_graphs.erdos_renyi_graph(n,param1)
    G_W = nx.Graph()
    for (u, v) in G.edges():
      G_W.add_weighted_edges_from([(u, v, 1)])

    G = copy.deepcopy(G_W)
    #print(G.nodes())
    #print(G.edges())
    ver_arr = [v for v in G.nodes()]
    ver_arr = ver_arr[:len(ver_arr)//2]
    Ts = [ver_arr]

    param = 2
    #G_S = greedy_spanner(G, param)
    G_S = top_down(G, Ts, multiplicative_check, param)[0]
    file_name_init = "graph_" + common_part_of_name + '_' + str(n) + '_' + str(i)
    file_name = file_name_init + '_train'
    edge_based_to_node_based(G, Ts, G_S, root_folder, file_name, True)

    while verify_spanner_with_checker(G_S, G, all_pairs_from_subset(ver_arr), multiplicative_check, param):
      # randomly remove an edge
      r = random.randint(0, len(G_S.edges())-1)
      edgs = [(u, v) for (u, v) in G_S.edges()]
      (u, v) = edgs[r]
      #print(edgs, r)
      G_S.remove_edge(u, v)
    file_name_init = "graph_" + common_part_of_name + '_' + str(n) + '_' + str(data_size+i)
    file_name = file_name_init + '_train'
    edge_based_to_node_based(G, Ts, G_S, root_folder, file_name, False)

    G = copy.deepcopy(G_W)
    #print(G.nodes())
    #print(G.edges())
    ver_arr = [v for v in G.nodes()]
    ver_arr = ver_arr[len(ver_arr)//2:]
    Ts = [ver_arr]

    param = 2
    #G_S = greedy_spanner(G, param)
    G_S = top_down(G, Ts, multiplicative_check, param)[0]
    file_name_init = "graph_" + common_part_of_name + '_' + str(n) + '_' + str(i)
    file_name = file_name_init + '_test'
    edge_based_to_node_based(G, Ts, G_S, root_folder, file_name, True)

    while verify_spanner_with_checker(G_S, G, all_pairs_from_subset(ver_arr), multiplicative_check, param):
      # randomly remove an edge
      r = random.randint(0, len(G_S.edges())-1)
      edgs = [(u, v) for (u, v) in G_S.edges()]
      (u, v) = edgs[r]
      #print(edgs, r)
      G_S.remove_edge(u, v)
    file_name_init = "graph_" + common_part_of_name + '_' + str(n) + '_' + str(data_size+i)
    file_name = file_name_init + '_test'
    edge_based_to_node_based(G, Ts, G_S, root_folder, file_name, False)

def edge_based_to_node_based(G, Ts, ST, ROOT_FOLDER, FILE_NAME, spanner):
  print(ST.nodes(), ST.edges())
  f2 = open(ROOT_FOLDER+'/'+FILE_NAME+"_node_weighted.txt", 'w')
  for n in ST.nodes():
    node_str = ""
    node_str += str(n) + ' '
    node_str += '0' + ' '
    if n in Ts[0]:
      node_str += "terminal" + ' ' + "in_solution"
      if spanner: node_str += ' ' + "spanner"
      else: node_str += ' ' + "not_spanner"
    else:
      fnd = False
      for u, v in ST.edges():
        if n==u or n==v:
          fnd = True
          node_str += "non_terminal" + ' ' + "in_solution"
          if spanner: node_str += ' ' + "spanner"
          else: node_str += ' ' + "not_spanner"
          break
      if not fnd:
        node_str += "non_terminal" + ' ' + "not_in_solution"
        if spanner: node_str += ' ' + "spanner"
        else: node_str += ' ' + "not_spanner"
    f2.write(node_str+"\n")
  for n in G.nodes():
    if n in ST.nodes(): continue
    node_str = ""
    node_str += str(n) + ' '
    node_str += '0' + ' '
    node_str += "non_terminal" + ' ' + "not_in_solution"
    if spanner: node_str += ' ' + "spanner"
    else: node_str += ' ' + "not_spanner"
    f2.write(node_str+"\n")
  edges_to_node = dict()
  #cntr = len(G.nodes())+1
  cntr = len(G.nodes())
  for u, v in G.edges():
    edges_to_node[str(u)+','+str(v)] = cntr
    node_str = ""
    node_str += str(cntr) + ' '
    node_str += str(G.get_edge_data(u,v)['weight']) + ' '
    fnd = False
    for x, y in ST.edges():
      if (x==u and y==v) or (x==v and y==u):
        fnd = True
        node_str += "non_terminal" + ' ' + "in_solution"
        if spanner: node_str += ' ' + "spanner"
        else: node_str += ' ' + "not_spanner"
        break
    if not fnd:
      node_str += "non_terminal" + ' ' + "not_in_solution"
      if spanner: node_str += ' ' + "spanner"
      else: node_str += ' ' + "not_spanner"
    f2.write(node_str+"\n")
    cntr += 1
  for u, v in G.edges():
    f2.write(str(u)+' '+str(edges_to_node[str(u)+','+str(v)])+"\n")
    f2.write(str(v)+' '+str(edges_to_node[str(u)+','+str(v)])+"\n")
    cntr += 1
  f2.close()

generate_graphs()

