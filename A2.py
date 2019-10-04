#copied from GCN github to load the dataset

import numpy as np
import pickle as pkl
import copy 
import networkx as nx
import igraph as ig
import scipy.sparse as sp
import community
from collections import Counter
from scipy.sparse.linalg.eigen.arpack import eigsh
from networkx.algorithms.community import LFR_benchmark_graph
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib
matplotlib.use('agg')
import pylab as plt
import sys


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("networks/real-node-label/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("networks/real-node-label/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # idx_test = test_idx_range.tolist()
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y)+500)

    # train_mask = sample_mask(idx_train, labels.shape[0])
    # val_mask = sample_mask(idx_val, labels.shape[0])
    # test_mask = sample_mask(idx_test, labels.shape[0])

    # y_train = np.zeros(labels.shape)
    # y_val = np.zeros(labels.shape)
    # y_test = np.zeros(labels.shape)
    # y_train[train_mask, :] = labels[train_mask, :]
    # y_val[val_mask, :] = labels[val_mask, :]
    # y_test[test_mask, :] = labels[test_mask, :]

    return adj, labels


'''
Load the synthetic LFR dataset
'''
def load_synthetic(n=250, tau1=3, tau2=1.5, mu=0.1):
    G = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5, min_community=20, seed=10)
    for n in G.nodes:
    	G.nodes[n]['value'] = list(G.nodes[n]['community'])[0]
    true_coms = list(nx.get_node_attributes(G,'value').values())
    com_keys = list(Counter(true_coms).keys())
    for i in range(0, len(true_coms)):
    	G.nodes[i]['value'] = com_keys.index(true_coms[i])

    true_labels = list(nx.get_node_attributes(G,'value').values())
    true_labels = [ int(x) for x in true_labels ]

    #remove self edges 
    selfE = list(G.selfloop_edges())
    for (i,j) in selfE:
        G.remove_edge(i,j)

    #convert all graph to undirected
    G = nx.Graph(G)
    nG = nx.Graph(G)

    # first convert the networkx graph to igraph 
    G = ig.Graph.Adjacency((nx.to_numpy_matrix(G) > 0).tolist())
    G.to_undirected()

    return (G, nG, true_labels)




'''
Load the real-classic dataset

dataset_str = 'strike', 'karate', 'polblogs', 'polbooks' or 'football'

'''
def load_real_classic(dataset_str):
    data_path = 'networks/real-classic/'
    if (dataset_str is 'karate'):
        G = nx.karate_club_graph()
        # use the ground truth in https://piratepeel.github.io/slides/groundtruth_presentation.pdf
        # label attribute is the ground truth in this case
        nx.set_node_attributes(G, 0, 'value')
        for n in G.nodes:
        	G.nodes[n]['value'] = G.nodes[n]['club']
        #turn categorical community labels into integer ones
        true_coms = list(nx.get_node_attributes(G,'value').values())
        com_keys = list(Counter(true_coms).keys())
        for i in range(0, len(true_coms)):
        	G.nodes[i]['value'] = com_keys.index(true_coms[i])


        # instructor_com = [0,1,2,3,4,5,6,7,10,11,12,13,16,17,19,21]
        # president_com = [8,9,14,15,18,20,22,23,24,25,26,27,28,29,30,31,32,33]
        # instructor_com = [x+1 for x in instructor_com]
        # president_com = [x+1 for x in president_com]
        # for i in president_com:
        #     G.nodes[i-1]['value'] = 1
    else:
        G = nx.read_gml(data_path+dataset_str+'.gml', label='id')

        if (dataset_str is 'polbooks'):
            #turn categorical community labels into integer ones
            true_coms = list(nx.get_node_attributes(G,'value').values())
            com_keys = list(Counter(true_coms).keys())
            for i in range(0, len(true_coms)):
                G.nodes[i]['value'] = com_keys.index(true_coms[i])

    #remove self edges 
    selfE = list(G.selfloop_edges())
    for (i,j) in selfE:
        G.remove_edge(i,j)
    true_labels = list(nx.get_node_attributes(G,'value').values())
    true_labels = [ int(x) for x in true_labels ]

    #convert all graph to undirected
    G = nx.Graph(G)

    # first convert the networkx graph to igraph 
    G = ig.Graph.Adjacency((nx.to_numpy_matrix(G) > 0).tolist())
    G.to_undirected()
    A = G.get_edgelist()
    nG = nx.Graph(A)
    return (G, nG, true_labels)


def load_real_node(dataset_str):

    (A, labels) = load_data(dataset_str)    
    labels = [ np.where(l == 1)[0][0] if len(np.where(l == 1)[0]) > 0 else 0  for l in labels]
    G = nx.Graph(A)
    for n in G.nodes:
        G.nodes[n]['value'] = labels[n]
    true_labels = list(nx.get_node_attributes(G,'value').values())
    true_labels = [ int(x) for x in true_labels ]
    # first convert the networkx graph to igraph 
    G = ig.Graph.Adjacency((nx.to_numpy_matrix(G) > 0).tolist())
    G.to_undirected()
    A = G.get_edgelist()
    nG = nx.Graph(A)
    return (G, nG, true_labels)




#https://python-louvain.readthedocs.io/en/latest/
'''
Input the graph

output the communities detected by the louvain algorithm
in the form of a list (community number )of lists (node idx of the community node)
'''
# def louvain_algorithm(G, name, plot=False):
#     partition = community.best_partition(G)
#     print (" there are " + str(len(partition)) + " communities detected by the louvain algorithm")
#     return partition

'''
Clauset-Newman-Moore greedy modularity maximization
https://networkx.github.io/documentation/latest/reference/algorithms/generated/networkx.algorithms.community.modularity_max.greedy_modularity_communities.html
Early method for modularity optimization
output: in the form of a list (community number )of lists (node idx of the community node)
'''
def greedy_modularity(G):
    partition = G.community_fastgreedy()
    clusters= partition.as_clustering()
    membership = [ord(c) if type(c) == str else c for c in clusters.membership]
    #print (" there are " + str(max(membership)) + " communities detected by the fastgreedy algorithm")
    return  membership


'''
igraph walktrap algorithm
https://igraph.org/python/doc/igraph.Graph-class.html#community_walktrap
expect igraph input
'''

def walktrap(G):
    partition = G.community_walktrap()
    clusters= partition.as_clustering()
    membership = [ord(c) if type(c) == str else c for c in clusters.membership]
    #print (" there are " + str(max(membership)) + " communities detected by the walktrap algorithm")
    return membership


'''
igraph
https://igraph.org/python/doc/igraph.Graph-class.html#community_infomap
'''
def infomap(G):
    partition = G.community_infomap()
    clusters= partition
    membership = [ord(c) if type(c) == str else c for c in clusters.membership]
    #print (" there are " + str(max(membership)) + " communities detected by the walktrap algorithm")
    return membership


def label_propagation(G):
    partition = G.community_label_propagation()
    clusters= partition
    membership = [ord(c) if type(c) == str else c for c in clusters.membership]
    #print (" there are " + str(max(membership)) + " communities detected by the walktrap algorithm")
    return membership





def evaluation(G, pred_labels, true_labels, name):
    Modularity = G.modularity(pred_labels)
    NMI = ig.compare_communities(pred_labels, true_labels, method='nmi')
    ARI = ig.compare_communities(pred_labels, true_labels, method='ari')
    return (Modularity, NMI, ARI)


def print_result(resultList, name, datasets):
    print ("the algorithm is " + name)
    idx = 0
    TMavg = 0
    TNavg = 0
    TAavg = 0

    for dataset in datasets:
        print (dataset)
        Mavg = sum(resultList[idx:idx+10][0]) / len(resultList[idx:idx+10][0])
        Navg = sum(resultList[idx:idx+10][1]) / len(resultList[idx:idx+10][1])
        Aavg = sum(resultList[idx:idx+10][2]) / len(resultList[idx:idx+10][2])
        print ("modularity is " + str(Mavg))
        print ("NMI is " + str(Navg))
        print ("ARI is " + str(Aavg))
        TMavg = TMavg + Mavg
        TNavg = TNavg + Navg
        TAavg = TAavg + Aavg
        idx = idx + 10
    print ("Overall : ")
    print ("modularity is " + str(TMavg/ len(datasets)))
    print ("NMI is " + str(TNavg/ len(datasets)))
    print ("ARI is " + str(TAavg/ len(datasets)))

    print ("----------------------------------------------------")
    print ("----------------------------------------------------")


def run_real_classic():
	#real_classic: 'strike', 'karate', 'polblog', 'polbooks' or 'footbal'
    real_classic = ['strike', 'karate', 'polblogs', 'polbooks', 'football']
    greedy_data = []
    walktrap_data = []
    infomap_data = []
    label_prop_data = []

    for datastr in real_classic:
        print ("looking at " + datastr + " dataset")
        (G, nG, true_labels) = load_real_classic(datastr)

        for i in range(0,10):
            pred_labels = greedy_modularity(G)
            (Modularity, NMI, ARI) = evaluation(G, pred_labels, true_labels, 'fast modularity')
            greedy_data.append([Modularity, NMI, ARI])

            pred_labels = walktrap(G)
            (Modularity, NMI, ARI) = evaluation(G, pred_labels, true_labels, 'walktrap')
            walktrap_data.append([Modularity, NMI, ARI])

            pred_labels = infomap(G)
            (Modularity, NMI, ARI) = evaluation(G, pred_labels, true_labels, 'infomap')
            infomap_data.append([Modularity, NMI, ARI])

            pred_labels = label_propagation(G)
            (Modularity, NMI, ARI) = evaluation(G, pred_labels, true_labels, 'label')
            label_prop_data.append([Modularity, NMI, ARI])


    print_result(greedy_data, "fast modularity", real_classic)
    print_result(walktrap_data, "walktrap", real_classic)
    print_result(infomap_data, "info", real_classic)
    print_result(label_prop_data, "label propagation", real_classic)








def run_real_node():
    real_node = ['cora','citeseer', 'pubmed']
    greedy_data = []
    walktrap_data = []
    infomap_data = []
    label_prop_data = []


    for datastr in real_node:
        print ("looking at " + datastr + " dataset")
        (G, nG, true_labels) = load_real_node(datastr)

        for i in range(0,10):
            pred_labels = greedy_modularity(G)
            (Modularity, NMI, ARI) = evaluation(G, pred_labels, true_labels, 'fast modularity')
            greedy_data.append([Modularity, NMI, ARI])

            pred_labels = walktrap(G)
            (Modularity, NMI, ARI) = evaluation(G, pred_labels, true_labels, 'walktrap')
            walktrap_data.append([Modularity, NMI, ARI])

            pred_labels = infomap(G)
            (Modularity, NMI, ARI) = evaluation(G, pred_labels, true_labels, 'infomap')
            infomap_data.append([Modularity, NMI, ARI])

            pred_labels = label_propagation(G)
            (Modularity, NMI, ARI) = evaluation(G, pred_labels, true_labels, 'label')
            label_prop_data.append([Modularity, NMI, ARI])

    print_result(greedy_data, "fast modularity", ['cora','citeseer', 'pubmed'])
    print_result(walktrap_data, "walktrap", ['cora','citeseer', 'pubmed'])
    print_result(infomap_data, "info", ['cora','citeseer', 'pubmed'])
    print_result(label_prop_data, "label propagation", ['cora','citeseer', 'pubmed'])



def run_synthetic():
    (G, nG, true_labels) = load_synthetic()
    datastr = 'LFR'
    print ("looking at " + datastr + " dataset")
    greedy_data = []
    walktrap_data = []
    infomap_data = []
    label_prop_data = []
    for i in range(0,10):
        pred_labels = greedy_modularity(G)
        (Modularity, NMI, ARI) = evaluation(G, pred_labels, true_labels, 'fast modularity')
        greedy_data.append([Modularity, NMI, ARI])

        pred_labels = walktrap(G)
        (Modularity, NMI, ARI) = evaluation(G, pred_labels, true_labels, 'walktrap')
        walktrap_data.append([Modularity, NMI, ARI])

        pred_labels = infomap(G)
        (Modularity, NMI, ARI) = evaluation(G, pred_labels, true_labels, 'infomap')
        infomap_data.append([Modularity, NMI, ARI])

        pred_labels = label_propagation(G)
        (Modularity, NMI, ARI) = evaluation(G, pred_labels, true_labels, 'label')
        label_prop_data.append([Modularity, NMI, ARI])

    print_result(greedy_data, "fast modularity", [datastr])
    print_result(walktrap_data, "walktrap", [datastr])
    print_result(infomap_data, "info", [datastr])
    print_result(label_prop_data, "label propagation", [datastr])

'''
Karate ground truth can be found online
The value file is the ground truth communities
Use ARI to find the best overall algorithm
'''

'''
Run 10 times for each dataset 
Report the average for real-classic, LFR and GCN dataset
Calculate modularity by igraph 
https://igraph.org/python/doc/igraph.Graph-class.html
'''

def main():
    #G = nx.erdos_renyi_graph(100, 0.05)
    run_real_node()
    run_real_classic()
    run_synthetic()








    

if __name__ == "__main__":
    main()



