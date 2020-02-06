import gnn.GNN as GNN
import gnn.gnn_utils
import Net as n

# Provide your own functions to generate input data
#inp, arcnode, nodegraph, labels = set_load()
from load import *
inp, arcnode, nodegraph, labels = loadmat('cli_10_5_200.mat')
print("******************************************")
print("inp", inp)
print("arcnode", arcnode)
print("nodegraph", nodegraph)
print("labels", labels)
input_dim = 10
state_dim = 5
output_dim = 200
num_epoch = 10
count = 100

# Create the state transition function, output function, loss function and  metrics
net = n.Net(input_dim, state_dim, output_dim)

# Create the graph neural network model
g = GNN.GNN(net, input_dim, output_dim, state_dim)

#Training

for j in range(0, num_epoch):
    g.Train(inp, arcnode, labels, count, nodegraph)

    # Validate
    print(g.Validate(inp_val, arcnode_val, labels_val, count, nodegraph_val))


