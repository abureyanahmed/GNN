import tensorflow as tf
import numpy as np

def weight_variable(shape, nm):
    # function to initialize weights
    initial = tf.truncated_normal(shape, stddev=0.1)
    tf.summary.histogram(nm, initial, collections=['always'])
    return tf.Variable(initial, name=nm)

class Net:
    # class to define state and output network

    def __init__(self, input_dim, state_dim, output_dim):
        # initialize weight and parameter

        self.EPSILON = 0.00000001

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.state_input = self.input_dim - 1 + state_dim  # removing the id_ dimension

        #### TO BE SET ON A SPECIFIC PROBLEM
        self.state_l1 = 30#[30]
        self.state_l2 = self.state_dim

        self.output_l1 = 20#[20]
        self.output_l2 = self.output_dim

    def netSt(self, inp):
        with tf.variable_scope('State_net'):

            layer1 = tf.layers.dense(inp, self.state_l1, activation=tf.nn.tanh)
            layer2 = tf.layers.dense(layer1, self.state_l2, activation=tf.nn.tanh)

            return layer2

    def netOut(self, inp):

            layer1 = tf.layers.dense(inp, self.output_l1, activation=tf.nn.tanh)
            layer2 = tf.layers.dense(layer1, self.output_l2, activation=tf.nn.softmax)

            return layer2

    def Loss(self, output, target, output_weight=None):
        # method to define the loss function
        #lo = tf.losses.softmax_cross_entropy(target, output)
        output = tf.maximum(output, self.EPSILON, name="Avoiding_explosions")  # to avoid explosions
        xent = -tf.reduce_sum(target * tf.log(output), 1)
        lo = tf.reduce_mean(xent)
        return lo

    def Metric(self, target, output, output_weight=None):
        # method to define the evaluation metric
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(target, 1))
        metric = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return metric


import scipy
import scipy.io as spio

sess = tf.Session()

import tensorflow as tf
import numpy as np
import  gnn.gnn_utils as gnn_utils
import gnn.GNN as GNN 


import networkx as nx
import scipy as sp

import matplotlib.pyplot as plt

##### GPU & stuff config
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

import os

#folder_name = "exp_BA_GNN_696E/"
#folder_name = "exp_ER_GNN_696E/"
folder_name = "exp_WS_GNN_696E/"
filenames = os.listdir(folder_name)

print("length:", len(filenames))

file_names = []
for fname in filenames:
  if fname[-3:]=="txt":
    file_names.append(fname)

print("graph file lengths:", len(file_names))

solution_folder = folder_name + "log_folder_exact/"
solution_files = os.listdir(solution_folder)

print("solution file length:", len(solution_files))

sol_file_names = []

for fname in file_names:
  sol_file_names.append("log_folder_exact/"+fname[:-4]+"_output.txt")

print("Graph files:", [fname for fname in file_names[:5]])
print("Solution graph files:", sol_file_names[:5])

folder_name = "./exp_BA_GNN_696E/"
#file_size = 200
file_size = 2560
file_names = []
sol_file_names = []
#"graph_BA_200_1_"+str(i)+".txt" for i in range(1, file_size)
for i in [20,40,60,80]:
  for j in list(range(10,210,10)):
    #for k in range(1,33):
    for k in range(0,32):
      file_names.append("graph_BA_200_1_"+str(i)+"_"+str(j)+'_'+str(k)+".txt")
      sol_file_names.append("log_folder_exact/graph_BA_200_1_"+str(i)+"_"+str(j)+'_'+str(k)+"_output.txt")
#file_size = 200
#sol_file_names = ["log_folder_exact/graph_BA_200_1_"+str(i)+"_output.txt" for i in range(1, file_size)]
#sol_file_names = ["log_folder_cmp/output_"+str(i)+".dat" for i in range(1, 2557)]

#yfolder_name = "./exp_BA_GNN_696E/"
#file_size = 150
#file_names = ["graph_BA_200_1_"+str(i)+".txt" for i in range(1, file_size)]
#sol_file_names = ["log_folder_exact/graph_BA_200_1_"+str(i)+"_output.txt" for i in range(1, file_size)]

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

def prepare_gnn_data(folder_name, file_names, sol_file_names):
  node_id_additive = 0
  total_edges = []
  total_nodes = []
  total_labels = []

  for i, (file_name, sol_file_name) in enumerate(zip(file_names, sol_file_names)):
    G = nx.Graph()
    edge_list, tree_ver = take_input(folder_name + file_name)
    for e in edge_list:
      G.add_edge(e[0], e[1])
    max_id = max([max(u, v) for u, v, _ in edge_list])
    min_id = min([min(u, v) for u, v, _ in edge_list])
    #edge_list = [[node_id_additive+u, node_id_additive+v, i] for u, v, _ in edge_list]
    edge_list1 = [[node_id_additive+u, node_id_additive+v, G.degree[u], G.degree[v], i] for u, v, _ in edge_list]
    edge_list2 = [[node_id_additive+v, node_id_additive+u, G.degree[v], G.degree[u], i] for u, v, _ in edge_list]
    #node_list = [[node_id_additive+u, 1] if u in tree_ver[0] else [node_id_additive+u, 0] for u in range(max_id+1)]
    node_list = [[node_id_additive+u, 1, i, G.degree[u]] if u in tree_ver[0] else [node_id_additive+u, 0, i, G.degree[u]] for u in range(max_id+1)]
    sol_edge_list, sol_tree_ver = take_input(folder_name + sol_file_name)
    #labels = [[0] for u in range(max_id+1)]
    labels = [[0, 1] for u in range(max_id+1)]
    for u, v, w in sol_edge_list:
      #labels[u] = [1]
      #labels[v] = [1]
      labels[u] = [1, 0]
      labels[v] = [1, 0]
    node_id_additive += (max_id+1)
    #total_edges += edge_list
    total_edges += edge_list1
    total_edges += edge_list2
    total_nodes += node_list
    total_labels += labels
  return total_edges, total_nodes, total_labels

#total_edges, total_nodes, total_labels = prepare_gnn_data(folder_name, file_names[:file_size-50], sol_file_names[:file_size-50])
total_edges, total_nodes, total_labels = prepare_gnn_data(folder_name, file_names[:file_size-500], sol_file_names[:file_size-500])

E_tot = np.asarray(total_edges)
N_tot = np.asarray(total_nodes)

labels = np.asarray(total_labels)
'''
labels_np = np.zeros((len(labels), 2))
for i in range(len(labels)):
  print(labels[i][0])
  labels_np[i][0] = labels[i][0]
  labels_np[i][1] = labels[i][1]
labels = labels_np
'''
print(labels[:100])

inp, arcnode, graphnode = gnn_utils.from_EN_to_GNN(E_tot, N_tot)

threshold = 0.01
learning_rate = 0.001#.001
state_dim = 5
tf.reset_default_graph()
input_dim = inp.shape[1]
#output_dim = 1
output_dim = 2
max_it = 100#50
#num_epoch = 5000
#num_epoch = 6000
num_epoch = 50
optimizer = tf.train.AdamOptimizer
# initialize state and output network
net = Net(input_dim, state_dim, output_dim)

param = "st_d" + str(state_dim) + "_th" + str(threshold) + "_lr" + str(learning_rate)
print(param)

tensorboard = True#False

g = GNN.GNN(net, input_dim, output_dim, state_dim,  max_it, optimizer, learning_rate, threshold, graph_based=False, param=param, config=config,
            tensorboard=tensorboard)

# train the model
count = 0

######
print(inp.shape)
print(labels.shape)
for j in range(0, num_epoch):
    #print("labels", labels)
    _, it = g.Train(inputs=inp, ArcNode=arcnode, target=labels, step=count)
    if count % 30 == 0:
        print("Epoch ", count)
        print("Training: ", g.Validate(inp, arcnode, labels, count))
        # end = time.time()
        # print("Epoch {} at time {}".format(j, end-start))
        # start = time.time()

    count = count + 1

met = g.Evaluate(inp, arcnode, labels)

pr = g.Predict(inp, arcnode, labels)
print(pr[0][:100])

#test_file_names = ["graph_ER_50_1_L_50_"+str(i)+".txt" for i in range(151, 200)]
#test_sol_file_names = ["log_folder_exact/graph_ER_50_1_L_50_"+str(i)+"_output.txt" for i in range(151, 200)]
#test_file_names = file_names[file_size-50:file_size]
#test_sol_file_names = sol_file_names[file_size-50:file_size]
test_file_names = file_names[file_size-500:file_size]
test_sol_file_names = sol_file_names[file_size-500:file_size]

test_total_edges, test_total_nodes, test_total_labels = prepare_gnn_data(folder_name, test_file_names, test_sol_file_names)

print("len(test_total_edges)", len(test_total_edges), "nodes", len(test_total_nodes))
E_tot_test = np.asarray(test_total_edges)
N_tot_test = np.asarray(test_total_nodes)
labels_test = np.asarray(test_total_labels)

inp_test, arcnode_test, graphnode_test = gnn_utils.from_EN_to_GNN(E_tot_test, N_tot_test)

met = g.Evaluate(inp_test, arcnode_test, labels_test)

pr = g.Predict(inp_test, arcnode_test, labels_test)
pr = [pr[0][i][0]-pr[0][i][1] for i in range(pr[0].shape[0])]
print(type(pr), "len(pr)", len(pr), "type(pr[0])", type(pr[0]), "pr[0].shape", pr[0].shape, "pr[1]", pr[1])
#np.savetxt("test_predictions.txt", pr, fmt='%s')

import numpy as np
import networkx as nx

def generate_solution_graphs(prediction, folder_name, file_names, sol_file_names):
 test_graph_edges, test_solution_edges, test_terminals = [], [], []
 connected_graphs = 0
 disconnected_graphs = 0
 disconnected_opt_solutions = 0
 rat_arr = []
 node_id_additive = 0

 for j, (file_name, sol_file_name) in enumerate(zip(file_names, sol_file_names)):
  edge_list, tree_ver = take_input(folder_name + file_name)
  sol_edge_list, sol_tree_ver = take_input(folder_name + sol_file_name)
  max_id = max([max(u, v) for u, v, _ in edge_list])
  min_id = min([min(u, v) for u, v, _ in edge_list])
  pred = pr[node_id_additive:node_id_additive+max_id+1]
  sorted_ind = np.argsort(pred)
  #print(pred)
  #print(sorted_ind)
  pred = np.round(pred)
  opt = len(sol_edge_list)
  G = nx.Graph()
  for e in edge_list:
   u, v, w = e
   G.add_edge(u, v)
  V = []
  '''
  for i, val in enumerate(pred):
   if val==1:
    V.append(i)
    #G.add_edge(*id_to_edge[i])
  '''
  for terminal in tree_ver[0]:
   if terminal not in V:
    V.append(terminal)
  S = G.subgraph(V)
  #'''
  #is_disconnected = False
  #if not nx.is_connected(S):
  # is_disconnected = True
  is_disconnected = not nx.is_connected(S)
  k = len(sorted_ind)-1
  #print("k", k)
  while not nx.is_connected(S):
   u = sorted_ind[k]
   if u not in V:
    V.append(u)
    S = G.subgraph(V)
   k -= 1
  #if is_disconnected:
  # mst=nx.minimum_spanning_edges(S)
  # apprx = len(list(mst))
  # if opt==apprx:
  #  disconnected_opt_solutions += 1
  mst=nx.minimum_spanning_edges(S)
  apprx = len(list(mst))
  if opt==apprx:
    disconnected_opt_solutions += 1
  #'''
  if nx.is_connected(S):
   connected_graphs += 1
   mst=nx.minimum_spanning_edges(S)
   apprx = len(list(mst))
   rat_arr.append(apprx/opt)
  else:
   disconnected_graphs += 1

  node_id_additive += (max_id+1)

 print("No. of connected graphs:", connected_graphs)
 total_rat = 0
 max_rat = 0
 min_rat = 1000
 for rat in rat_arr:
  max_rat = max(max_rat, rat)
  min_rat = min(min_rat, rat)
  total_rat += rat
 avg_rat = total_rat/connected_graphs
 print("maximum ratio:", max_rat)
 print("average ratio:", avg_rat)
 print("minimum ratio:", min_rat)
 print("No. of disconnected graphs:", disconnected_graphs)
 print("No. of disconnected graphs become optimal after connecting:", disconnected_opt_solutions)

#generate_solution_graphs(pr_arr, folder_name, test_file_names, test_sol_file_names)
generate_solution_graphs(pr, folder_name, test_file_names, test_sol_file_names)



