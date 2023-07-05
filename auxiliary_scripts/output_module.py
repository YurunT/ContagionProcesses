import json
import matplotlib.pyplot as plt
from collections import defaultdict 
import numpy as np
from datetime import datetime
import time
import sys, os
sys.path.append(os.path.abspath("."))
from global_vars import *

def div(x, y):
    if y == 0:
        return 0
    else:
        return x*1.0/y
    
def get_key(val, my_dict):
    for key, value in my_dict.items():
         if val == value:
            return key
    
def get_change_name(change):
    assert change in change_metrics_dict.values()
    return 'change_' + get_key(change, change_metrics_dict)

        

def generate_path(path_name):
    if not os.path.exists(path_name):
        os.makedirs(path_name)

def json_save(file_name, file):
    with open(file_name, 'w') as fp:
        json.dump(file, fp)
        
def get_para_setting_str(paras):
    print("From output module")
    para_setting_str = 'm' + str(paras.m) + '_T' + str(paras.T)
    for idx in range(len(paras.tm1)):
        tmi1 = 'tm%d,1' %(idx + 1) + '_' + '{0:.2f}'.format(paras.tm1[idx]) 
        tmi2 = 'tm%d,2' %(idx + 1) + '_' + '{0:.2f}'.format(paras.tm2[idx])
        para_setting_str += tmi1
        para_setting_str += '_'
        para_setting_str += tmi2
    return para_setting_str

def get_common_path(paras):
    change_folder = get_change_name(paras.change)
    common_path = paras.modelname + '/' + paras.itemname + '/' + change_folder + '/' + paras.msg + '/' + get_para_setting_str(paras) + '/'
    return common_path

def get_setting_path(paras, time_exp):
    setting_path = base_path + 'simulation/' + get_common_path(paras) + 'n' + str(paras.n) + '_ttle' + str(paras.e) + '/' + time_exp + '/Settings/' 
    print("Setting_path:", setting_path)
    return setting_path

def get_exp_path(paras, cp, mean_degree, time_exp, start_strain):
    exp_path = base_path + 'simulation/' + get_common_path(paras) + 'n' + str(paras.n) + '_ttle' + str(paras.e) + '/' + time_exp +'/ss'+ str(start_strain) + '/meandegree'+ str(mean_degree) +'/' +'cp' + str(cp)
    print("Experiment path:", exp_path)
    return exp_path

def get_analysis_path(paras, time_analysis,):
    analysis_path = base_path + 'analysis/'   + get_common_path(paras) + time_analysis
    print("Analysis path:", analysis_path)
    return analysis_path

def get_jsonable_dict(unjsonable_dict, mean_degree_list):
    jsonable_infection_size = dict()
    for k, v in unjsonable_dict.items(): # k: mask type i, v: dict, key: md1_md2
        type_k_res = defaultdict(dict)
        for md1_md2, res in v.items(): 
            md1, md2 = md1_md2.split('_')
            type_k_res[md1][md2] = res
        jsonable_infection_size[k] = type_k_res
    return jsonable_infection_size
    
def write_analysis_results(paras, infection_size, mean_degree_list, is_ray=False):
    ''' Save the results for anaysis.
        Analysis code are accelarated by parellel
    '''

    print("Analysis finished! Start wrting json...")
   
    ######### Generate paths ########## 
    time_analysis = datetime.now().strftime("%m%d%H:%M")
    analysis_path = get_analysis_path(paras, time_analysis,)
    res_path = analysis_path + '/' + 'Results'
    setting_path = analysis_path + '/' + 'Settings'

    generate_path(analysis_path)
    generate_path(setting_path)
    generate_path(res_path)

    ######### Save results ########## 
    json_save(setting_path + "/paras.json",vars(paras))
    if is_ray:
        infection_res = transfer_ray_list(infection_size)
    else:
        infection_res = get_jsonable_dict(infection_size, mean_degree_list)
    json_save(res_path + "/infection_res.json", infection_res)
    np.save(setting_path + '/mean_degree_list.npy', mean_degree_list)
    
def transfer_ray_list(results):
    '''
    Transfer the results given by analysis script using ray to the dict
    '''
    infection_size = dict()

    for typei in range(len(results[0][0])):
        indiv = defaultdict(dict)
        ttl = defaultdict(dict)
        for md, ES, ttlES in results:

            indiv[md[0]][md[1]] = ES[typei]
            ttl[md[0]][md[1]] = ttlES
        infection_size[typei] = indiv
        infection_size['ttl'] = ttl
        
    return infection_size
    
    
def write_cp_raw_results(results, start_strain, mean_degree, cp, time_exp, start_time, paras,):
    ''' Save the checkponint raw results for simulation.
        Sim codes are accelarated by Ray.
    '''
    ######### Generate paths ########## 
    ExpPath = get_exp_path(paras, cp, mean_degree, time_exp, start_strain)
    generate_path(ExpPath)

    ######### Save results ########## 
    json_save(ExpPath + '/results.json', results)
    
    ######### Time setting ########## 
    now_finish = datetime.now() # current date and time
    timeExp = now_finish.strftime("%m%d%H:%M")
    print("checkpoint %d Done! for exp %s" %(cp, timeExp))
    print("--- %.2s seconds ---" % (time.time() - start_time))

def write_exp_settings(time_exp, paras, mean_degree_list):
    setting_path = get_setting_path(paras, time_exp)
    generate_path(setting_path)
    json_save(setting_path + 'paras.json', vars(paras))
    np.save(setting_path + 'mean_degree_list.npy', np.array(mean_degree_list))