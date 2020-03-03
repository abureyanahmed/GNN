import networkx as nx
from input_functions import *
from networkx.algorithms.approximation import steiner_tree
map_file = "experiment_1/id_to_file.csv"

f = open(map_file, 'r')
while True:
  l = f.readline()
  arr = l.split(';')
  if len(arr)<2:
    break
  CODE_FILE = arr[1]
  ROOT_FOLDER = arr[2]
  FILE_NAME = arr[3]
  OUTPUT_FILE = arr[4]
  print(ROOT_FOLDER, FILE_NAME)
  filename = ROOT_FOLDER + '/' + FILE_NAME +'.txt'
  G, Ts = build_networkx_graph(filename)
  ST = steiner_tree(G, Ts[0])
  print(ST.nodes(), ST.edges())
  f2 = open(ROOT_FOLDER+'/'+FILE_NAME+"_node_weighted.txt", 'w')
  for n in ST.nodes():
    node_str = ""
    node_str += str(n) + ' '
    node_str += '0' + ' '
    if n in Ts[0]:
      node_str += "terminal" + ' ' + "in_solution"
    else:
      fnd = False
      for u, v in ST.edges():
        if n==u or n==v:
          fnd = True
          node_str += "non_terminal" + ' ' + "in_solution"
          break
      if not fnd:
        node_str += "non_terminal" + ' ' + "not_in_solution"
    f2.write(node_str+"\n")
  edges_to_node = dict()
  cntr = len(G.nodes())+1
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
        break
    if not fnd:
      node_str += "non_terminal" + ' ' + "not_in_solution"
    f2.write(node_str+"\n")
    cntr += 1
  for u, v in G.edges():
    f2.write(str(u)+' '+str(edges_to_node[str(u)+','+str(v)])+"\n")
    f2.write(str(v)+' '+str(edges_to_node[str(u)+','+str(v)])+"\n")
    cntr += 1
  f2.close()
f.close()
