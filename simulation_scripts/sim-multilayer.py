import random
import sys, pdb
import site, os
import collections
import numpy as np
import igraph as ig
from datetime import datetime
import time
import multiprocessing
from multiprocessing import Manager
import ray
sys.path.append(os.path.abspath("../auxiliary_scripts/"))
# sys.path.append(os.path.abspath("../analysis_scripts/"))
from input_module import *
from output_module import *
from tnn import generate_new_transmissibilities_mask
from main_aux import *
# from degree_distributions import get_powerlaw_with_expcutoff_pk
from scipy.stats import poisson
import numpy as np
import math
from mpmath import polylog

# ray.shutdown()
# ray.init()

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def div(x, y):
    if y == 0:
        return 0
    else:
        return x*1.0/y

def create_network(mean_degree, num_nodes):
#     degree_sequence = np.random.poisson(mean_degree, num_nodes)
    kmax = 1000
    pk_list = get_powerlaw_with_expcutoff_pk(kmax)
    degree_sequence = np.random.choice(a=np.arange(0, kmax+1), size=(num_nodes), p=pk_list)
    while (np.sum(degree_sequence) % 2 !=0):
#         degree_sequence = np.random.poisson(mean_degree, num_nodes)
        degree_sequence = np.random.choice(a=np.arange(0, kmax+1), size=(num_nodes), p=pk_list)
    return ig.Graph.Degree_Sequence(list(degree_sequence)).simplify()
    
def infected_rule(infected_neighbors_dict, T_list, susceptible_nodes, num_strain, mask_prob, mask_status, g):
    new_nodes_list = [set() for i in range(num_strain)]
    
    if len(infected_neighbors_dict.keys()) != 0:
        for susceptible_node in infected_neighbors_dict:
            trial_list = infected_neighbors_dict[susceptible_node] # All of his infected neighbors
            random.shuffle(trial_list)
            infected = False
            for infected_neighbor, infected_neighbor_type in trial_list:
                if infected:
                    break
                strain_type_idx = mask_status[susceptible_node]
                
                for e in g.es.select(_source=infected_neighbor, _target=susceptible_node):
                    if e['color'] == 'blue':
                        T = T_list[0][infected_neighbor_type][mask_status[susceptible_node]]
                    else:
                        T = T_list[1][infected_neighbor_type][mask_status[susceptible_node]]

                    if random.random() < T: # susceptible node is infected
                        susceptible_nodes.remove(susceptible_node)
                        new_nodes_list[strain_type_idx].add(susceptible_node)
                        infected = True
                        break
                        
    return new_nodes_list, susceptible_nodes

def get_accumulated_mask_probs(mask_probs,):
    '''
    Input: 
    mask_probs: A list of mask probabilities, where sum(mask_probs) = 1. 
    Output: 
    accumulated_mask_probs: An length 1 interval with each segment having the length of each ele in mask prob
    e.g. 
    mask_probs = [0.3, 0.2, 0.1, 0.4]
    accumulated_mask_probs = [0, 0.3, 0.5, 0.6, 1]
    '''
    accumulated_mask_probs = [0]
    
    for idx, m in enumerate(mask_probs):
        accumulated_mask_probs.append(m + accumulated_mask_probs[idx])
    
    return accumulated_mask_probs

def get_node_status(mask_roll_dice, accumulated_mask_probs):
    '''
    Input:  accumulated_mask_probs
    Output: int, corresponding to the idx of the mask_prob, 
            the maks wearing type of a single node.
    '''
    
    mask_status = []
    for roll_dice in mask_roll_dice:
        distance = roll_dice - np.array(accumulated_mask_probs)

        mask_type = -1
        for idx, dis in enumerate(distance):
            if dis >= 0 and distance[idx + 1] < 0:
                mask_type = idx
                break
        mask_status.append(mask_type)
        
    return mask_status

def get_mask_status(mask_probs, num_nodes):
    '''
    Input:  
    mask_prob: A list of mask probabilities(len(mask_prob = num_mask_types)
    num_nodes: Graph size
    Output: 
    mask_status: A list of mask wearing states for each node in the graph.
    '''
    accumulated_mask_probs = get_accumulated_mask_probs(mask_probs,)
    mask_roll_dice = np.random.rand(num_nodes)
    mask_status = get_node_status(mask_roll_dice, accumulated_mask_probs)
        
    return mask_status

def get_seed(start_strain, num_nodes, mask_status, num_strain):
    if start_strain < 0 or start_strain >= num_strain:
        print("sim-ray.py: start_strain out of index!")
        assert False
        
    seed = int(np.random.randint(0, num_nodes - 1))   
    strain_list = [set() for i in range(num_strain)]
    while mask_status[seed] != start_strain:
        seed = int(np.random.randint(0, num_nodes - 1))
    strain_list[start_strain] = set([seed])
    
    return seed, strain_list
    

def evolution(g, T_list, mask_prob, start_strain):
    node_set = set(g.vs.indices)
    num_nodes = len(node_set)
    num_strain = len(T_list[0]) # num_mask_types
    mask_status = get_mask_status(mask_prob, num_nodes, )
    seed, strain_list = get_seed(start_strain, num_nodes, mask_status, num_strain) 
    susceptible_nodes = node_set
    
    for strain_set in strain_list:
        susceptible_nodes = susceptible_nodes.difference(strain_set)
    new_nodes_list = strain_list # level L - 1
    
    while(sum([len(new_nodes) for new_nodes in new_nodes_list])):
        neighbor_dict = collections.defaultdict(list) # susceptible nodes in level L, its parents are in the list

        for strain_type, strain_set in enumerate(new_nodes_list): # string type == 0: wear mask
            strain_neighbors_list = []
            
            for infected_node in strain_set:
                strain_neighbors_list = g.neighbors(infected_node)
                for susceptible_node in strain_neighbors_list:
                    if susceptible_node not in susceptible_nodes: continue
                    neighbor_dict[susceptible_node].append((infected_node, strain_type))
                
        new_nodes_list, susceptible_nodes = infected_rule(neighbor_dict, T_list, susceptible_nodes, num_strain, mask_prob, mask_status, g) # Get next level

        strain_list = [strain_list[s_idx].union(s) for s_idx, s in enumerate(new_nodes_list)]

    num_infected = sum([len(s) for s in strain_list])
    num_infected_list = map(len, strain_list)
    return num_infected, num_infected_list

# def get_pk(k):
#         '''Create a model function for a powerlaw distribution with exponential cutoff.
#         This function credits to https://pyepydemic.readthedocs.io/en/latest/cookbook/population-powerlaw-cutoff.html

#         :param alpha: the exponent of the distribution; alpha here is lambda_ in Cojoining Speeds up... 
#         :param kappa: the degree cutoff; kappa here is the gamma in Cojoining Speeds up... 
#         :returns: a model function'''
#         alpha=2.5
#         kappa = 10
#         C = polylog(alpha, math.exp(-1.0 / kappa))
        
#         if k == 0: # Only change we made is pk(0) = 0
#             return 0
#         else:
#             return float((pow((k + 0.0), -alpha) * math.exp(-(k + 0.0) / kappa)) / C)

def get_powerlaw_with_expcutoff_pk(kmax):
    pks = []
    alpha=2.5
    kappa=10
#     C = polylog(alpha, math.exp(-1.0 / kappa))
    C = 1.1477157068692658
#     print('Good here')
    for k in range(kmax + 1):
        if k == 0:
            pks.append(0)
        else:
            pks.append(float((pow((k + 0.0), -alpha) * math.exp(-(k + 0.0) / kappa)) / C))     
    return np.array(pks)

@ray.remote
def runExp(i, mean_degree, T_list, start_type, paras):
    primary_net = create_network(mean_degree[0], paras.n)
    primary_net.es['color'] = 'blue'

    
    n_second = int(paras.alpha * paras.n)
    second_net = create_network(mean_degree[1], n_second)
    second_net.es['color'] = 'red'

    second_edge_list = []
    for edge in second_net.es:
        second_edge_list.append((edge.source, edge.target))
    primary_net.add_edges(second_edge_list, attributes={'color': 'red'})
    
    size, size_list = evolution(primary_net, T_list, paras.m, start_type)
    return div(size,paras.n), [div(sizei, paras.n) for sizei in size_list]

def main():
    ########### Get commandline input ###########
    paras = parse_args(sys.argv[1:])
    paras_check(paras)
    mean_degree_list = get_mean_degree_list(paras)
    k_max, T_list = resolve_paras(paras)
    
    ############ Start Exp ############
    now = datetime.now() # current date and time
    start_time = time.time()
    time_exp = now.strftime("%m%d%H:%M")
    print("-------Exp start at:" + time_exp + '-------')

    for start_type in range(len(paras.m)):
        ray.shutdown()
        ray.init()
        for mean_degree in mean_degree_list:
            for cp in range(1, int(paras.e/paras.cp) + 1): # cp order
                results_ids = []
                for i in range(paras.cp):
                    results_ids.append(runExp.remote(i, mean_degree, T_list, start_type, paras),)  
                results = ray.get(results_ids)
                write_cp_raw_results(results, start_type, mean_degree, cp, time_exp, start_time, paras,)

        write_exp_settings(time_exp, paras, mean_degree_list)
        ray.shutdown()

    now_finish = datetime.now() # current date and time
    print("All Done! for:" + time_exp)
    print("--- %.2s seconds in total ---" % (time.time() - start_time))
main()
ray.shutdown()