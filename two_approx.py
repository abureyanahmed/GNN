from os import walk
import numpy as np
import networkx as nx

# read graph

def is_comment(x):
 if x[0]=='#':
  return True
 return False

def take_input(input_file):
 file = open(input_file,"r")
 #print("File name: "+input_file)
 while True:
  l = file.readline()
  #print(l)
  if not is_comment(l):
   break
 m = int(l)
 edge_list = list()
 for i in range(m):
    while True:
     l = file.readline()
     if len(l) == 0:
      break
     if not is_comment(l):
      break
    t_arr1 = []
    t_arr2 = l.split()
    if(len(t_arr2)<3):break
    t_arr1.append(int(t_arr2[0])-1)
    t_arr1.append(int(t_arr2[1])-1)
    #t_arr1.append(int(t_arr2[0]))
    #t_arr1.append(int(t_arr2[1]))
    #t_arr1.append(int(t_arr2[2]))
    t_arr1.append(float(t_arr2[2]))
    edge_list.append(t_arr1)

 levels = int(file.readline())
 tree_ver=[]
 #tree_ver = [(int(x)-1) for x in raw_input().split()]
 for l in range(levels):
  #print "Steiner tree vertices of level "+str(l+1)+":"
  tree_ver.append([(int(x)-1) for x in file.readline().split()])
  #tree_ver.append([(int(x)) for x in file.readline().split()])

 file.close()
 return edge_list, tree_ver

def get_graph_info(train_graph_folder, train_solution_folder):
 f = []
 for (dirpath, dirnames, filenames) in walk(train_graph_folder):
  f.extend(filenames)
  break
 train_edges, train_solutions, train_terminals = [], [], []
 for file in f:
  graph_edges, terminals = take_input(train_graph_folder+file)
  solution_edges, terminals = take_input(train_solution_folder+file[:-4]+"_output.txt")
  train_edges.append(graph_edges)
  train_solutions.append(solution_edges)
  train_terminals.append(terminals)
 return train_edges, train_solutions, train_terminals

def ordered_edge(u, v):
  if u>v:
    t = u
    u = v
    v = t
  return u, v

def edge_to_number(labels = 'nodes'):
 #max_node_size = 20
 max_node_size = 50
 edge_to_id = dict()
 id_to_edge = dict()
 idx = 0
 for u in range(max_node_size):
  for v in range(u+1, max_node_size):
   edge_to_id[(u, v)] = idx
   id_to_edge[idx] = (u, v)
   idx = idx + 1
 n_inputs = idx + max_node_size
 #n_inputs = idx + 2*max_node_size
 #n_inputs = 2*idx + 2*max_node_size
 n_outputs = max_node_size
 if labels == 'nodes':
  return n_inputs, n_outputs, idx, edge_to_id, id_to_edge
 else:
  return n_inputs, idx, idx, edge_to_id, id_to_edge

def generate_dataset(train_graph_edges, train_terminals, train_solution_edges, labels = 'nodes'):
 n_inputs, n_outputs, max_edge_index, edge_to_id, id_to_edge = edge_to_number(labels)
 train_features = []
 train_labels = []
 for i, graph_edges in enumerate(train_graph_edges):
  feature_arr = [0 for j in range(n_inputs)]
  label_arr = [0 for j in range(n_outputs)]
  for e in graph_edges:
   u, v, w = e
   u, v = ordered_edge(u, v)
   feature_arr[edge_to_id[(u, v)]] = 1
  '''
  G = nx.Graph()
  for e in graph_edges:
   u, v, w = e
   G.add_edge(u, v, weight=w)
  length = dict(nx.all_pairs_shortest_path_length(G))
  k = 0
  for l in range(G.number_of_nodes()):
   for j in range(l+1, G.number_of_nodes()):
    feature_arr[max_edge_index+k] = length[l][j]
    k += 1
  '''
  terminals = train_terminals[i]
  for t in terminals[0]:
   feature_arr[max_edge_index+t] = 1
  '''
  for u in G.nodes():
   feature_arr[max_edge_index+G.number_of_nodes()+u] = G.degree[u]
  '''
  solution_edges = train_solution_edges[i]
  for e in solution_edges:
   for e in solution_edges:
    u, v, w = e
    u, v = ordered_edge(u, v)
    if labels=='nodes':
     #'''
     label_arr[u] = 1
     label_arr[v] = 1
     #'''
     '''
     if u not in terminals[0]:
      label_arr[u] = 1
     if v not in terminals[0]:
      label_arr[v] = 1
     '''
    else:
     label_arr[edge_to_id[(u, v)]] = 1

  train_features.append(feature_arr)
  train_labels.append(label_arr)

 train_features = np.vstack(train_features)
 train_labels = np.vstack(train_labels)
 return train_features, train_labels, n_inputs, n_outputs

def load_data(folder_name):
 #folder_name = "/Users/abureyanahmed/GNN/TSP_Steiner/Steinertree-master/steiner_dataset/"
 #folder_name = "/Users/abureyanahmed/GNN/TSP_Steiner/Steinertree-master/steiner_dataset50/"
 train_graph_folder = folder_name + "train/graph/"
 train_solution_folder = folder_name + "train/solution/"
 test_graph_folder = folder_name + "test/graph/"
 test_solution_folder = folder_name + "test/solution/"

 train_graph_edges, train_solution_edges, train_terminals = get_graph_info(train_graph_folder, train_solution_folder)
 test_graph_edges, test_solution_edges, test_terminals = get_graph_info(test_graph_folder, test_solution_folder)

 return train_graph_edges, train_solution_edges, train_terminals, test_graph_edges, test_solution_edges, test_terminals

def Kruskal(G):

    MST=nx.create_empty_copy(G); # MST(G)
    N=nx.number_of_nodes(G)
    E=nx.number_of_edges(G)
    i=0; # counter for edges of G
    k=0; # counter for MST(G)

    edge_list = sorted(G.edges(data=True), key=lambda x:x[2]['weight'])

    while k<(N-1) and i<(E):
        e=edge_list[i];
        i+=1
        if not nx.has_path(MST,e[0],e[1]):
            MST.add_edge(e[0],e[1],weight=e[2]['weight'])
            k+=1

    return(MST)

def SteinerTree(G,T):

    HG=nx.Graph()
    HG.add_nodes_from(T)  # Hyper graph with nodes T and edges with weight equal to distance
    n=len(T)

    for i in range(n):
        for j in range(i+1,n):
            HG.add_edge(T[i], T[j], weight=nx.shortest_path_length(G,T[i], T[j],'weight'))

    HG_MST = Kruskal(HG)

    G_ST=nx.Graph()
    for e in HG_MST.edges(data=False):
        P=nx.shortest_path(G,e[0],e[1],'weight')
        G_ST.add_path(P)

    # find the minimum spanning tree of the resultant graph

    return(G_ST)

folder_name = "./exp_ER_GNN_Steiner_TSP_50/"
train_graph_edges50, train_solution_edges50, train_terminals50, test_graph_edges50, test_solution_edges50, test_terminals50 = load_data(folder_name)
print("50 nodes training size:", len(train_graph_edges50))
print("50 nodes testing size:", len(test_graph_edges50))
train_graph_edges, train_solution_edges, train_terminals, test_graph_edges, test_solution_edges, test_terminals = train_graph_edges50, train_solution_edges50, train_terminals50, test_graph_edges50, test_solution_edges50, test_terminals50

total_apprx = 0
total_opt = 0
total_rat = 0
for i, graph_edges in enumerate(test_graph_edges):
  G = nx.Graph()
  for e in graph_edges:
    G.add_edge(e[0], e[1], weight=1)
  G_sol = nx.Graph()
  for e in test_solution_edges[i]:
    G_sol.add_edge(e[0], e[1], weight=1)
  ST = SteinerTree(G, test_terminals[i][0])
  total_apprx += ST.number_of_edges()
  total_opt += G.number_of_edges()
  total_rat += (ST.number_of_edges()/G_sol.number_of_edges())
print("Avg ratio", total_rat/len(test_graph_edges))

total_apprx = 0
total_opt = 0
total_rat = 0
for i, graph_edges in enumerate(train_graph_edges):
  G = nx.Graph()
  for e in graph_edges:
    G.add_edge(e[0], e[1], weight=1)
  G_sol = nx.Graph()
  for e in train_solution_edges[i]:
    G_sol.add_edge(e[0], e[1], weight=1)
  ST = SteinerTree(G, train_terminals[i][0])
  total_apprx += ST.number_of_edges()
  total_opt += G.number_of_edges()
  cur_rat = (ST.number_of_edges()/G_sol.number_of_edges())
  total_rat += cur_rat
  if cur_rat!=1:
    print(cur_rat)
print("Avg ratio", total_rat/len(train_graph_edges))


