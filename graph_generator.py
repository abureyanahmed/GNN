import networkx as nx
import random
import sys
import math
if len(sys.argv) < 8:
        print("usage: python3 graph_generator.py"
                + "\n" +"levels node_size path_to_graph_file={/Users/abureyanahmed/Desktop/testenv/venv/Graph\ generator/name}"
                + "\n" + "class_of_graph={0/watts_strogatz,1/erdos_renyi,2/preferential,3/geometric} param1 param2"
                + "\n" + "node_distribution_in_levels={0/linear,1/exponential}"
                + "\n" + "watts_strogatz:param1->neighbors,param2->probability"
                + "\n" + "erdos_renyi:param1->probability"
                + "\n" + "preferential:param1->neighbors"
                + "\n" + "geometric:param1->Distance threshold value")
        print()
        quit()

#graph = nx.fast_gnp_random_graph(10,.2)
levels = int(sys.argv[1])
size = int(sys.argv[2])
param1 = float(sys.argv[5])
param2 = float(sys.argv[6])
node_distribution_in_levels = int(sys.argv[7])
nodes = size
avg_deg = 0
if size<10:
        avg_deg = 3
else:
        #avg_deg = 6
        avg_deg = param1
class_of_graph = int(sys.argv[4])
while True:
        if class_of_graph==0:
                #graph = nx.connected_watts_strogatz_graph(nodes,avg_deg,.2)
                graph = nx.connected_watts_strogatz_graph(nodes,int(avg_deg),param2)
        elif class_of_graph==1:
                graph = nx.generators.random_graphs.erdos_renyi_graph(nodes,param1)
        elif class_of_graph==2:
                graph = nx.generators.random_graphs.barabasi_albert_graph(nodes,int(param1))
        elif class_of_graph==3:
                graph = nx.generators.geometric.random_geometric_graph(nodes,param1)
        if nx.is_connected(graph):
                break
print("For steiner app:")
print(graph.number_of_edges())
# used below also, copy and replace print with write
if class_of_graph==3:
        pos=nx.get_node_attributes(graph,'pos')
edges = graph.edges()
edge_weights = []
j = 0
for e in edges:
        edge_weights.append(random.randint(1,10))
        print(str(e[0]+1)+" "+str(e[1]+1)+" "+str(edge_weights[j]))
        j = j+1
        #if class_of_graph!=3:
        #       print(str(e[0]+1)+" "+str(e[1]+1)+" "+str(random.randint(1,10)))
        #else:
        #       x1, y1 = pos[e[0]]
        #       x2, y2 = pos[e[1]]
        #       #print(str(e[0]+1)+" "+str(e[1]+1)+" "+str(math.pow((x1-x2)**2+(y1-y2)**2,.5)))
        #       print(str(e[0]+1)+" "+str(e[1]+1)+" 1")
print(str(levels))
for l in range(levels):
        if node_distribution_in_levels==0:
                if size<10:
                        steiner_nodes = size-2
                else:
                        steiner_nodes = int(nodes*(l+1)/(levels+1))
                steiner_nodes_str = ""
                for j in range(steiner_nodes-1):
                        #steiner_nodes_str = steiner_nodes_str + str(j+1) + " "
                        steiner_nodes_str = steiner_nodes_str + str(j) + " "
                #steiner_nodes_str = steiner_nodes_str + str(steiner_nodes);
                steiner_nodes_str = steiner_nodes_str + str(steiner_nodes-1);
        elif node_distribution_in_levels==1:
                if size<10:
                        steiner_nodes = size-2
                else:
                        steiner_nodes = int(math.ceil(nodes*1.0/math.pow(2,levels-l)))
                steiner_nodes_str = ""
                for j in range(steiner_nodes-1):
                        #steiner_nodes_str = steiner_nodes_str + str(j+1) + " "
                        steiner_nodes_str = steiner_nodes_str + str(j) + " "
                #steiner_nodes_str = steiner_nodes_str + str(steiner_nodes);
                steiner_nodes_str = steiner_nodes_str + str(steiner_nodes-1);
        print(steiner_nodes_str)
#print(stretch_factor)
file = open(sys.argv[3]+"_train.txt","w")
file_test = open(sys.argv[3]+"_test.txt","w")
file.write(str(graph.number_of_edges())+"\n");
file_test.write(str(graph.number_of_edges())+"\n");
if class_of_graph==3:
        pos=nx.get_node_attributes(graph,'pos')
edges = graph.edges()
j = 0
for e in edges:
        #file.write(str(e[0]+1)+" "+str(e[1]+1)+" "+str(edge_weights[j])+"\n")
        file.write(str(e[0])+" "+str(e[1])+" "+str(edge_weights[j])+"\n")
        file_test.write(str(e[0])+" "+str(e[1])+" "+str(edge_weights[j])+"\n")
        j = j+1
        #if class_of_graph!=3:
        #       file.write(str(e[0]+1)+" "+str(e[1]+1)+" "+str(random.randint(1,10))+"\n")
        #else:
        #       x1, y1 = pos[e[0]]
        #       x2, y2 = pos[e[1]]
        #       #file.write(str(e[0]+1)+" "+str(e[1]+1)+" "+str(math.pow((x1-x2)**2+(y1-y2)**2,.5))+"\n")
        #       file.write(str(e[0]+1)+" "+str(e[1]+1)+" 1\n")
file.write(str(levels)+"\n")
file_test.write(str(levels)+"\n")
for l in range(levels):
        if node_distribution_in_levels==0:
                if size<10:
                        steiner_nodes = size-2
                else:
                        steiner_nodes = int(nodes*(l+1)/(levels+1))
                steiner_nodes_str = ""
                steiner_nodes_str_test = ""
                for j in range(steiner_nodes-1):
                        #steiner_nodes_str = steiner_nodes_str + str(j+1) + " "
                        steiner_nodes_str = steiner_nodes_str + str(j) + " "
                        steiner_nodes_str_test = steiner_nodes_str + str(nodes-1-j) + " "
                #steiner_nodes_str = steiner_nodes_str + str(steiner_nodes);
                steiner_nodes_str = steiner_nodes_str + str(steiner_nodes-1);
                steiner_nodes_str_test = steiner_nodes_str + str(nodes-1-steiner_nodes+1);
        elif node_distribution_in_levels==1:
                if size<10:
                        steiner_nodes = size-2
                else:
                        steiner_nodes = int(math.ceil(nodes*1.0/math.pow(2,levels-l)))
                steiner_nodes_str = ""
                steiner_nodes_str_test = ""
                for j in range(steiner_nodes-1):
                        #steiner_nodes_str = steiner_nodes_str + str(j+1) + " "
                        steiner_nodes_str = steiner_nodes_str + str(j) + " "
                        steiner_nodes_str_test = steiner_nodes_str + str(nodes-1-j) + " "
                #steiner_nodes_str = steiner_nodes_str + str(steiner_nodes);
                steiner_nodes_str = steiner_nodes_str + str(steiner_nodes-1);
                steiner_nodes_str_test = steiner_nodes_str + str(nodes-1-steiner_nodes+1);
        file.write(steiner_nodes_str+"\n")
        file_test.write(steiner_nodes_str_test+"\n")
#file.write(stretch_factor+"\n")
file.close()
file_test.close()
