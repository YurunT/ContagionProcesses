import multiprocessing
import sys, os
import numpy as np
sys.path.append(os.path.abspath("."))
from tnn import *
from global_vars import *
    
def get_mean_degree_list(paras):
    '''
    Given [mind1, maxd1], ns1 and [mind2, maxd2], ns2
    Return all the combinations of the (md1, md2) tuples
    
    For example, paras.mind = [0, 2], paras.maxd = [1, 4], paras.ns = [2, 3]
    mean_degree_list1 = np.linspace(0, 1, 2) = [0, 1]
    mean_degree_list2 = np.linspace(2, 4, 3) = [2, 3, 4]
    
    return [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]
    '''
    if paras.mdl1 is not None: 
        mean_degree_list1 = paras.mdl1
    else:
        mean_degree_list1 = np.linspace(paras.mind[0], paras.maxd[0], paras.ns[0])
    
    if paras.mdl2 is not None: 
        mean_degree_list2 = paras.mdl2
    else:
        mean_degree_list2 = np.linspace(paras.mind[1], paras.maxd[1], paras.ns[1])
        
    mean_degree_lists = []
    for md1 in mean_degree_list1:
        for md2 in mean_degree_list2:
            mean_degree_lists.append((md1, md2))
    return mean_degree_lists

def paras_check(paras):
    if paras.tm1 is None:
        print("------------ CMD INPUT INVALID ------------")
        print("Error: NO tm1 specified!")
        print("Please specify list of OUTward effeciencies!")
        print("------------ CMD INPUT INVALID ------------")
        assert False
    if paras.tm2 is None:
        print("------------ CMD INPUT INVALID ------------")
        print("Error: NO tm2 specified!")
        print("Please specify list of INward effeciencies!")
        print("------------ CMD INPUT INVALID ------------")
        assert False
    if len(paras.tm2) != len(paras.tm1):
        print("------------ CMD INPUT INVALID ------------")
        print("Error: len(tm1) %d not matching len(tm2) %d!" %(len(paras.tm1), len(paras.tm2)))
        print("Please specify correct num of in/out-ward effeciencies!")
        print("------------ CMD INPUT INVALID ------------")
        assert False
    if sum(paras.m) < 0.99:
        print("------------ CMD INPUT INVALID ------------")
        print("Error: sum(m) should equals 1!")
        print("Your m is:", paras.m)
        print("Your m sum up to:", sum(paras.m))
        print("Please check m list!")
        print("------------ CMD INPUT INVALID ------------")
        assert False
#     if len(paras.tm1) != len(paras.m):
#         print("------------ CMD INPUT INVALID ------------")
#         print("Error: len(tm1) %d not matching len(m) %d!" %(len(paras.tm1), len(paras.m)))
#         print("Please specify correct num of out/in-ward effeciencies and mask probs!")
#         print("------------ CMD INPUT INVALID ------------")
#         assert False
    if paras.modelname not in model_names:
        print("------------ CMD INPUT INVALID ------------")
        print("Error: NO model named %s!" %paras.modelname)
        print("Please select from", model_names)
        print("------------ CMD INPUT INVALID ------------")
        assert False
    if paras.itemname not in item_names:
        print("------------ CMD INPUT INVALID ------------")
        print("Error: NO analyzed item named %s!" %paras.itemname)
        print("Please select from", item_names)
        print("------------ CMD INPUT INVALID ------------")
        assert False
    if paras.change not in change_metrics_dict.values():
        print("------------ CMD INPUT INVALID ------------")
        print("Error: NO change metric as %s!" %paras.change)
        print("Please select numbers from", change_metrics_dict)
        print("------------ CMD INPUT INVALID ------------")
        assert False
    if paras.kmax and len(paras.kmax) != 2:
        print("------------ CMD INPUT INVALID ------------")
        print("Error: kmax need to have 2 values!")
        print("------------ CMD INPUT INVALID ------------")
        assert False
    if paras.kmax and (paras.kmax[0] < paras.maxd[0] or paras.kmax[1] < paras.maxd[1]):
        print("------------ CMD INPUT INVALID ------------")
        print("Error: kmax value has to be at least same with maxd!")
        print("------------ CMD INPUT INVALID ------------")
        assert False
    if paras.T and len(paras.T) != 2:
        print("------------ CMD INPUT INVALID ------------")
        print("Error: len(T) has to be 2!")
        print("------------ CMD INPUT INVALID ------------")
        assert False
    if  not paras.mdl1 and not paras.mdl2 and paras.mind and paras.maxd and (len(paras.mind) != 2 or len(paras.maxd) != 2) :
        print("------------ CMD INPUT INVALID ------------")
        print("Error: len(mind) and len(maxd) has to be 2!")
        print("------------ CMD INPUT INVALID ------------")
        assert False
    if not paras.mdl1 and not paras.mdl2 and paras.mind > paras.maxd:
        print("------------ CMD INPUT INVALID ------------")
        print("Error: mind should be smaller than maxd")
        print("------------ CMD INPUT INVALID ------------")
        assert False
    if  paras.maxd and paras.mind and paras.ns is None:
        print("------------ CMD INPUT INVALID ------------")
        print("Error: When specifying mean degree list using maxd and mind, needs to specify ns!")
        print("------------ CMD INPUT INVALID ------------")
        assert False    
    if not paras.mdl1 and not paras.mdl2 and paras.ns and len(paras.ns)!= 2:
        print("------------ CMD INPUT INVALID ------------")
        print("Error: need to specify ns for both networks!")
        print("------------ CMD INPUT INVALID ------------")
        assert False
       
           
def resolve_paras(paras):
    num_cores = min(paras.nc,multiprocessing.cpu_count())
    rho = 1.0 / paras.n
    if not paras.kmax:
        k_max = list(np.array(paras.maxd) * 5)
    else:
        k_max = paras.kmax
    tmask_list = get_tmask_list(paras)
    
    n_layers = len(paras.T)
    T_lists = []
    for layer_i in range(n_layers):
        T_lists.append(generate_new_transmissibilities_mask(tmask_list, paras.T[layer_i],))
    
    return k_max,T_lists

def resolve_paras_single(paras):
    num_cores = min(paras.nc,multiprocessing.cpu_count())
    rho = 1.0 / paras.n
#     k_max = paras.maxd * 5
    k_max = paras.kmax
#     print("k_max:", k_max)
#     k_max = 20
    tmask_list = get_tmask_list(paras)
    T_list = generate_new_transmissibilities_mask(tmask_list, paras.T,)
    
    if paras.modelname == 'mutation':
        q_dict, mu_dict = generate_new_transmissibilities_mutation(tmask_list, paras.T, paras.m)
        Q_list  = list(q_dict.values())
        mu_list = list(mu_dict.values())    
    if paras.modelname == 'mutation':
        return rho, k_max, T_list, Q_list, mu_list
    else:
        return k_max, T_list