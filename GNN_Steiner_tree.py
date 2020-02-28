import tensorflow as tf
import numpy as np
import gnn.gnn_utils as gnn_utils
import gnn.GNN as GNN
import Net_Simple as n

import networkx as nx
import scipy as sp

import matplotlib.pyplot as plt

##### GPU & stuff config
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

############# data creation ################

# GRAPH #1

# List of edges in the first graph - last column is the id of the graph to which the arc belongs
e = [[0, 1, 0], [1,2, 0], [2, 3, 0], [3, 4, 0], [2, 11, 0], [11, 6, 0], [0, 5, 0], [5, 6, 0], [6, 7, 0], [7, 4, 0], [6, 12, 0], [12, 9, 0], [0, 8, 0], [8, 9, 0], [9, 10, 0], [10, 4, 0]]
# undirected graph, adding other direction
e.extend([[i, j, num] for j, i, num in e])
#reorder
e = sorted(e)
E = np.asarray(e)


#number of nodes
edges = 13
# creating node features - simply one-hot values
N = np.eye(edges, dtype=np.float32)

W = np.zeros((13, 10), dtype=np.float32)
for i in range(13):
    W[i][0]=1
W[5][9]=1
W[5][0]=0
W[12][9]=1
W[12][0]=0

T = np.zeros((13, 1), dtype=np.float32)
T[0][0]=1
T[4][0]=1

# adding column thta represent the id of the graph to which the node belongs
N = np.concatenate((N, W),  axis=1 )
N = np.concatenate((N, T),  axis=1 )
N = np.concatenate((N, np.zeros((edges,1), dtype=np.float32)),  axis=1 )


# visualization graph
def plot_graph(E, N):
    g = nx.Graph()
    g.add_nodes_from(range(N.shape[0]))
    g.add_edges_from(E[:, :2])
    nx.draw(g, cmap=plt.get_cmap('Set1'), with_labels=True)
    plt.show()


plot_graph(E,N)



# GRAPH #2

# List of edges in the second graph - last column graph-id
e1 = [[0, 1, 1], [1,2, 1], [2, 3, 1], [3, 4, 1], [2, 11, 1], [11, 6, 1], [0, 5, 1], [5, 6, 1], [6, 7, 1], [7, 4, 1], [6, 12, 1], [12, 9, 1], [0, 8, 1], [8, 9, 1], [9, 10, 1], [10, 4, 1]]
# undirected graph, adding other direction
e1.extend([[i, j, num] for j, i, num in e1])
# reindexing node ids based on the dimension of previous graph (using unique ids)
e2 = [[a + N.shape[0], b + N.shape[0], num] for a, b, num in e1]
#reorder
e2 = sorted(e2)


edges_2 = 13


# Plot second graph

E1 = np.asarray(e1)

N1 = np.eye(edges_2,  dtype=np.float32)

W = np.zeros((13, 10), dtype=np.float32)
for i in range(13):
    W[i][0]=1
W[5][9]=1
W[5][0]=0
W[12][9]=1
W[12][0]=0

T = np.zeros((13, 1), dtype=np.float32)
T[2][0]=1
T[9][0]=1

N1 = np.concatenate((N1, W),  axis=1 )
N1 = np.concatenate((N1, T),  axis=1 )
N1 = np.concatenate((N1, np.zeros((edges_2,1), dtype=np.float32)),  axis=1 )

plot_graph(E1,N1)

E = np.concatenate((E, np.asarray(e2)), axis=0)

#N_tot = np.eye(edges + edges_2,  dtype=np.float32)
#N_tot = np.concatenate((N_tot, np.zeros((edges + edges_2,1), dtype=np.float32)),  axis=1 )
N_tot = np.zeros((N.shape[0]+N1.shape[0], N.shape[1]))
for i in range(N.shape[0]):
    for j in range(N.shape[1]):
        N_tot[i][j] = N[i][j]
        N_tot[i+13][j] = N1[i][j]


# Create Input to GNN

inp, arcnode, graphnode = gnn_utils.from_EN_to_GNN(E, N_tot)
labels = np.zeros(26, dtype=int)
labels[0]=1
labels[1]=1
labels[2]=1
labels[3]=1
labels[4]=1
labels[13+2]=1
labels[13+11]=1
labels[13+6]=1
labels[13+12]=1
labels[13+9]=1


labels = np.eye(max(labels)+1, dtype=np.int32)[labels]  # one-hot encoding of labels


################################################################################################
################################################################################################
################################################################################################
################################################################################################

# set input and output dim, the maximum number of iterations, the number of epochs and the optimizer
threshold = 0.01
learning_rate = 0.01
state_dim = 5
tf.reset_default_graph()
input_dim = inp.shape[1]
output_dim = labels.shape[1]
max_it = 50
num_epoch = 10000
optimizer = tf.train.AdamOptimizer

# initialize state and output network
net = n.Net(input_dim, state_dim, output_dim)

# initialize GNN
param = "st_d" + str(state_dim) + "_th" + str(threshold) + "_lr" + str(learning_rate)
print(param)

tensorboard = False

g = GNN.GNN(net, input_dim, output_dim, state_dim,  max_it, optimizer, learning_rate, threshold, graph_based=False, param=param, config=config,
            tensorboard=tensorboard)

# train the model
count = 0

######

for j in range(0, num_epoch):
    _, it = g.Train(inputs=inp, ArcNode=arcnode, target=labels, step=count)

    if count % 30 == 0:
        print("Epoch ", count)
        print("Training: ", g.Validate(inp, arcnode, labels, count))

        # end = time.time()
        # print("Epoch {} at time {}".format(j, end-start))
        # start = time.time()

    count = count + 1


# evaluate on the test set
# print("\nEvaluate: \n")
# print(g.Evaluate(inp_test[0], arcnode_test[0], labels_test, nodegraph_test[0])[0])

# GRAPH #1

# List of edges in the first graph - last column is the id of the graph to which the arc belongs
e = [[0, 1, 0], [1,2, 0], [2, 3, 0], [3, 4, 0], [2, 11, 0], [11, 6, 0], [0, 5, 0], [5, 6, 0], [6, 7, 0], [7, 4, 0], [6, 12, 0], [12, 9, 0], [0, 8, 0], [8, 9, 0], [9, 10, 0], [10, 4, 0]]
# undirected graph, adding other direction
e.extend([[i, j, num] for j, i, num in e])
#reorder
e = sorted(e)
E = np.asarray(e)


#number of nodes
edges = 13
# creating node features - simply one-hot values
N = np.eye(edges, dtype=np.float32)

W = np.zeros((13, 10), dtype=np.float32)
for i in range(13):
    W[i][0]=1
W[5][9]=1
W[5][0]=0
W[12][9]=1
W[12][0]=0

T = np.zeros((13, 1), dtype=np.float32)
T[6][0]=1
T[9][0]=1

# adding column thta represent the id of the graph to which the node belongs
N = np.concatenate((N, W),  axis=1 )
N = np.concatenate((N, T),  axis=1 )
N = np.concatenate((N, np.zeros((edges,1), dtype=np.float32)),  axis=1 )


inp_test, arcnode_test, graphnode_test = gnn_utils.from_EN_to_GNN(E, N)

labels_test = np.zeros(13, dtype=int)
labels_test[6]=1
labels_test[7]=1
labels_test[4]=1
labels_test[10]=1
labels_test[9]=1

labels_test = np.eye(max(labels_test)+1, dtype=np.int32)[labels_test]

print("\nEvaluate: \n")
print(g.Evaluate(inp_test, arcnode_test, labels_test)[0])

