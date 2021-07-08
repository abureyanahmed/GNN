import os
import sys
import random
import subprocess
import networkx as nx

experiment_folder = sys.argv[1]
id_file_name = sys.argv[2]
id_file_name = experiment_folder + id_file_name
log_folder_name = sys.argv[3]
GNN_folder_name = sys.argv[4]
folder_name = GNN_folder_name
#data_size = int(sys.argv[5])
data_size_start = int(sys.argv[5])
data_size_end = int(sys.argv[6])
#id_file_name = "/Users/abureyanahmed/Graph_spanners/exp_ER_wgt_20/id_to_file.csv"
#experiment_folder = "/Users/abureyanahmed/Graph_spanners/exp_ER_wgt_20/"
#folder_name = "./exp_all_GNN_spanner/"
'''
os.mkdir(folder_name)
os.mkdir(folder_name+"train/")
os.mkdir(folder_name+"train/graph/")
os.mkdir(folder_name+"train/solution/")
os.mkdir(folder_name+"test/")
os.mkdir(folder_name+"test/graph/")
os.mkdir(folder_name+"test/solution/")
quit()
'''
subprocess.run(["mkdir", folder_name])
subprocess.run(["mkdir", folder_name+"train/"])
subprocess.run(["mkdir", folder_name+"train/graph/"])
subprocess.run(["mkdir", folder_name+"train/solution/"])
subprocess.run(["mkdir", folder_name+"test/"])
subprocess.run(["mkdir", folder_name+"test/graph/"])
subprocess.run(["mkdir", folder_name+"test/solution/"])

#def parse_id_csv(experiment_folder, graph_type):
def parse_id_csv(experiment_folder):
  ids = []
  folders = []
  stretch = []
  file_names = []
  level = []
  nlevel = []
  node = []
  #f = open(experiment_folder + 'id_to_file.csv', 'r')
  f = open(id_file_name, 'r')
  line_number = 1
  while True:
   line = f.readline()
   if line=='':
    break
   arr = line.split(';')
   FILE_NAME = arr[3]
   #if not graph_type in FILE_NAME:
   # continue
   file_names.append(FILE_NAME)
   ids.append(arr[0])
   CODE_FILE = arr[1]
   ROOT_FOLDER = arr[2]
   folders.append(ROOT_FOLDER)
   '''
   if len(arr)>4 and arr[4]!='\n':
    STRETCH_FACTOR = float(arr[4])
    stretch.append(str(STRETCH_FACTOR))
   '''
   line_number += 1
  f.close()
  '''
  for i in range(len(folders)):
   node.append(int(file_names[i].split('_')[5]))
   level.append(file_names[i].split('_')[4])
   nlevel.append(int(file_names[i].split('_')[3]))
  return ids, folders, stretch, file_names, level, node, nlevel
  '''
  return ids, folders, stretch, file_names

from os import walk

f = []
'''
f1 = []
f2 = []
'''
#for (dirpath, dirnames, filenames) in walk("./exp_ER_GNN_Steiner_TSP_50/"):
#    f.extend(filenames)
#    break
#ids, folders, stretch, file_names, level, node, nlevel = parse_id_csv(experiment_folder, "ER")
ids, folders, stretch, file_names = parse_id_csv(experiment_folder)
#file_names = file_names[:data_size]
file_names = file_names[data_size_start:data_size_end]
#for i, nl in enumerate(nlevel):
for i, nl in enumerate(file_names):
  f.append(file_names[i])
'''
  #if nl==1:
  if level[i]=='E':
    f1.append(file_names[i])
  else:
    f2.append(file_names[i])
f = f1[:-40] + f2[:-40] + f1[-40:] + f2[-40:]
'''
#print(f)
#print(len(f))
#quit()

random.shuffle(f)
#print(f)

#folder_name = "./exp_ER_GNN_Steiner_TSP_50/"
file_names = []
sol_file_names = []
for elm in f:
 if "graph" in elm:
  file_names.append(experiment_folder+elm+".txt")
  sol_file_names.append(experiment_folder+log_folder_name+"/"+elm+"_output.txt")
#print(file_names)
#print(len(file_names))
#print(sol_file_names)
#quit()

test_size = int(.2*len(file_names))
#print("test_size", test_size)
if test_size==0:
 print("No test data!")
 quit()

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

def create_file_with_features(src_file, dest_file):
 edge_list, tree_ver = take_input(src_file)
 G = nx.Graph()
 for e in edge_list:
  u, v, w = e
  G.add_edge(u, v, weight=w)
 f = open(dest_file, 'w')
 #f.write("Node, degree\n")
 f.write(str(G.number_of_nodes()) + "\n")
 for u in G.nodes():
  f.write(str(u) + ' ' + str(G.degree[u]) + '\n')
 #f.write("Edge source, target, weight\n")
 f.write(str(len(list(G.edges()))) + "\n")
 for e in G.edges():
  u, v = e
  w = G[u][v]['weight']
  f.write(str(u) + ' ' + str(v) + ' ' + str(w) + '\n')
 #f.write("Source, target, distance\n")
 f.write(str(G.number_of_nodes()*(G.number_of_nodes()-1)/2) + '\n')
 length = dict(nx.all_pairs_shortest_path_length(G))
 nodes = list(G.nodes())
 for i in range(len(nodes)):
  for j in range(i+1, len(nodes)):
   u, v = nodes[i], nodes[j]
   d = length[u][v]
   f.write(str(u) + ' ' + str(v) + ' ' + str(d) + '\n')
 f.write(str(len(tree_ver)) + '\n')
 for terminals in tree_ver:
  s = str(terminals[0])
  for t in terminals[1:]:
   s = s + ' ' + str(t)
  f.write(s + '\n')
 f.close()

from shutil import copyfile

for i,fname in enumerate(file_names[:-test_size]):
 sol_src = sol_file_names[i]
 if os.path.exists(sol_src):
  src = file_names[i]
  dst = folder_name+"train/graph/"+file_names[i].split('/')[-1]
  #copyfile(src, dst)
  create_file_with_features(src, dst)
  #dst = folder_name+"train/solution/"+sol_file_names[i].split('/')[-1][:-11]+".txt"
  sol_dst = folder_name+"train/solution/"+sol_file_names[i].split('/')[-1]
  copyfile(sol_src, sol_dst)
#print(file_names[:-test_size])

for i,fname in enumerate(file_names[-test_size:]):
 sol_src = sol_file_names[-test_size:][i]
 if os.path.exists(sol_src):
  src = file_names[-test_size:][i]
  dst = folder_name+"test/graph/"+file_names[-test_size:][i].split('/')[-1]
  #copyfile(src, dst)
  create_file_with_features(src, dst)
  #dst = folder_name+"test/solution/"+sol_file_names[i].split('/')[-1][:-11]+".txt"
  sol_dst = folder_name+"test/solution/"+sol_file_names[-test_size:][i].split('/')[-1]
  copyfile(sol_src, sol_dst)
#print(file_names[-test_size:])


