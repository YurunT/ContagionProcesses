import re
import sys, os
from os import listdir
import collections
sys.path.append(os.path.abspath("."))
from global_vars import *
from output_module import *
from para_settings import *
from main_aux import *
from collections import defaultdict
import numpy as np

def test_import_aux():
    print("This is jupyter aux")
    
def separate_number_chars(s):
    res = re.split('([-+]?\d+\.\d+)|([-+]?\d+)', s.strip())
    res_f = [r.strip() for r in res if r is not None and r.strip() != '']
    return res_f
    
def json_load(json_path):
    with open(json_path) as json_file:
        item = json.load(json_file)
    return item

def get_time(exp_time, path):
    if exp_time == '':
        exp_times = [f for f in listdir(path) if f != '.ipynb_checkpoints']
        exp_time = max(exp_times) # Get the latest results
        print(exp_time)
    return exp_time

def get_mdl(path):
    mean_degrees = [f.split('meandegree')[1] for f in listdir(path) if f != '.ipynb_checkpoints']
    return mean_degrees
    
def get_ordered_values_by_key(x):
    ordered_values_list = list({k: v for k, v in sorted(x.items(), key=lambda item: item[1])}.values())
    return ordered_values_list

def get_parasobj(m=0.45, T=0.6, tm1=[0.3,1], tm2=[0.7,1], kmax=[20, 20], maxd=[10,10], mind=[9,9], ns=[2, 2], alpha=0.5, mdl1=None, mdl2=None, msg='test', modelname='mask', itemname='es', change_metric='m', n=50, e=10, checkpoint=5):
    paras = para_setting()
    paras.m = m
    paras.T = T
    paras.tm1 = tm1
    paras.tm2 = tm2
    paras.msg = msg
    paras.modelname = modelname
    paras.itemname = itemname
    paras.change = change_metrics_dict[change_metric]
    paras.n = n
    paras.e = e
    paras.cp = checkpoint
    paras.mind = mind
    paras.maxd = maxd
    paras.ns = ns
    paras.kmax = kmax
    paras.alpha = alpha
    paras.mdl1 = mdl1
    paras.mdl2 = mdl2
    paras_check(paras)
    return paras

def load_sim_settings(paras, time_exp):
    setting_path = get_setting_path(paras, time_exp)
    paras_arg = json_load(setting_path + 'paras.json')
    mean_degree_list = np.load(setting_path + 'mean_degree_list.npy')
    return paras_arg, mean_degree_list

def load_sim_raw_results(m=0.45, T=0.6, tm1=0.3, tm2=0.7, kmax=[20,20], maxd=[10, 10], mind=[9,9], ns=[2, 2], alpha=0, mdl1=None, mdl2=None, msg='test', modelname='mask', change_metric='m', n=50, e=10, checkpoint=5, time_exp='',):
    '''Load Simulation Results'''
  
    # Prepare paras obj
    infection_size = dict()
    itemname='es' # es and pe are using the same script for simulation
    paras = get_parasobj(m=m, T=T, tm1=tm1, tm2=tm2, kmax=kmax, maxd=maxd, mind=mind, ns=ns, alpha=alpha, mdl1=mdl1, mdl2=mdl2, msg=msg, modelname=modelname, itemname='es', change_metric=change_metric, n=n, e=e, checkpoint=checkpoint)
    paras.print_paras()
    print("time_exp: ", time_exp)
    
    # Prepare paths
    path = base_path + 'simulation/' + get_common_path(paras) + 'n' + str(paras.n) + '_ttle' + str(paras.e) + '/' 
    time_exp = get_time(time_exp, path)
    mdl_path = path + time_exp +'/ss'+ str(0) + '/'
#     mean_degree_list = get_mdl(mdl_path)
    mean_degree_list = get_mean_degree_list(paras)
    
    raw = defaultdict(dict)
    for start_strain in range(len(m)): # to accomodate the error of the running script
        for mean_degree in mean_degree_list:
            raw[start_strain][mean_degree] = dict()
            for cp in range(1, int(paras.e/paras.cp) + 1):
                print("cp:", cp)
                exp_path = get_exp_path(paras, cp, mean_degree, time_exp, start_strain)
                try:
                    raw[start_strain][mean_degree][cp] = json_load(exp_path + '/results.json') # results has paras.cp exp results
                except Exception as error:
                    print(error)
                    raw[start_strain][mean_degree][cp] = np.nan
                    
    
    paras_arg, mean_degree_list = load_sim_settings(paras, time_exp)
    
    res = dict()
    res['raw'] = raw
    res['paras'] = paras_arg
    res['mdl'] = mean_degree_list
    return res

def process_sim_cp(results, checkpoint, thr, num_strain):    
    # Get how many points are above thr within a checkpoint
    ttlEpidemicsSize = 0
    ttlnumEpidemics = 0
    EpidemicsSizePerStr = [0 for i in range(num_strain)] 
    
    fractionDic = dict()
    infectedPerStDic = dict()
    
    for ii in range(checkpoint):
        fractionDic[ii] = results[ii][0] ### ttl results
        infectedPerStDic[ii] = results[ii][1] ### individual strain results
        
        if fractionDic[ii] >= thr:
            ttlnumEpidemics += 1
            ttlEpidemicsSize += fractionDic[ii]

            for strain in range(num_strain):
                EpidemicsSizePerStr[strain] += infectedPerStDic[ii][strain]

    return ttlnumEpidemics, ttlEpidemicsSize, EpidemicsSizePerStr

def process_raw_GSCC(exp3_GSCC, n, num_strain, md, checkpoint_size):
    n_checkpoint = n / checkpoint_size
    GSCC_res = []

    for start_strain in range(num_strain):
        res_dict = dict()
        esGSCC_ttl = []
        esGSCC_0 = []
        esGSCC_1 = []

        peGSCC_avg = []
        peGSCC_0 = []
        peGSCC_1 = []
        for cp in range(int(n_checkpoint)):
            for ii in range(checkpoint_size):
                es_GSCC = exp3_GSCC['raw'][start_strain][md][int(cp)+1][ii][0]
                pe_GSCC = exp3_GSCC['raw'][start_strain][md][int(cp)+1][ii][1]

                esGSCC_ttl.append(es_GSCC['ttl'])
                esGSCC_0.append(es_GSCC['0'])
                esGSCC_1.append(es_GSCC['1'])

                peGSCC_avg.append(pe_GSCC['ttl'])
                peGSCC_0.append(pe_GSCC['0'])
                peGSCC_1.append(pe_GSCC['1'])

        res_dict['pe0'] = np.mean(peGSCC_0)
        res_dict['pe1'] = np.mean(peGSCC_1)
        res_dict['pe'] = np.mean(peGSCC_avg)

        res_dict['es0'] = np.mean(esGSCC_0)
        res_dict['es1'] = np.mean(esGSCC_1)
        res_dict['es'] = np.mean(esGSCC_ttl)

        GSCC_res.append(res_dict)
    return GSCC_res

def process_raw(raw, paras, thr,):
    # collect results for the entire exp (multiple checkpoints together)
    processed_res_strains = []
    num_strain = len(raw.keys())
    
    for start_strain, results in raw.items():
        processed_res = defaultdict(dict)
        
        for mean_degree, result in results.items():
            ttl_numepidemics = 0
            ttl_epidemicsize = 0
            ttl_EpidemicsSizePerStr = np.array([0.0 for i in range(num_strain)])
        
            
            for cp, cp_res in result.items():
                numEpidemics_cp, EpidemicsSize_cp, EpidemicsSizePerStr_cp = process_sim_cp(cp_res, paras['cp'], thr, num_strain)
                ttl_numepidemics += numEpidemics_cp
                ttl_epidemicsize += EpidemicsSize_cp
                ttl_EpidemicsSizePerStr += np.array(EpidemicsSizePerStr_cp)

            processed_res['es'][mean_degree] = div(ttl_epidemicsize * 1.0, ttl_numepidemics)
            processed_res['pe'][mean_degree] = div(ttl_numepidemics * 1.0, paras['e'])
        
            for strain in range(num_strain):
                processed_res['es%d' %strain][mean_degree] = div(ttl_EpidemicsSizePerStr[strain] * 1.0, ttl_numepidemics)
        
        ordered_res = dict()
        ordered_res['pe']  = get_ordered_values_by_key(processed_res['pe'])
        ordered_res['es']  = get_ordered_values_by_key(processed_res['es'])
        
        for strain in range(num_strain):
            ordered_res['es%d' %strain] = get_ordered_values_by_key(processed_res['es%d' %strain])
        
        processed_res_strains.append(ordered_res)
        
    return processed_res_strains
        
                
    
def load_analysis_results(m=0.45, T=0.6, tm1=0.3, tm2=0.7, kmax=20, maxd=[11, 9], mind=[10, 8], ns=[2, 2], alpha=0.5, mdl1=None, mdl2=None, msg='test', modelname='mask', itemname='pe', change_metric='m', time_analysis='', verbose=True):
    '''Load Analysis Results'''

    # Prepare paras obj
    infection_size = dict()
    paras = get_parasobj(m, T, tm1, tm2, kmax, maxd, mind, ns, alpha, mdl1, mdl2, msg, modelname, itemname, change_metric)
    
    if verbose:
        paras.print_paras()
        print("time_analysis: ", time_analysis)
    
    # Prepare paths
    path = base_path + 'analysis/' + get_common_path(paras)
    time_analysis = get_time(time_analysis, path)
    analysis_path = get_analysis_path(paras, time_analysis,)
    res_path = analysis_path + '/' + 'Results'
    setting_path = analysis_path + '/' + 'Settings'
    
    # Load results
    paras_json = json_load(setting_path + "/paras.json")
    infection_res = json_load(res_path + "/infection_res.json")
    mean_degree_list = np.load(setting_path + '/mean_degree_list.npy')
    infection_res['paras'] = paras_json
    infection_res['mdl'] = mean_degree_list 
    return infection_res


##### Figrure function #####
def get_range_str(mdl):
    range_str = "[%.2f, %.2f]" %(min(mdl), max(mdl))
    return range_str

def append_legend_list(legend_list, mdl, sim_or_analysis):
    if sim_or_analysis == 'sim':
        first_word = 'Sim'
    elif sim_or_analysis == 'analysis':
        first_word = 'Theory'
    else:
        assert False
        
    range_str = get_range_str(mdl)
    
    legend_list.append(first_word + "(mask) "   + range_str)
    legend_list.append(first_word + "(nomask) " + range_str)
    legend_list.append(first_word + "(total) "  + range_str)
    
def plot_anaylsis(res, ax, legend_list, marker='--'):

    ax.plot(res['mdl'], np.array(res['0']), 'g' + marker)
    ax.plot(res['mdl'], np.array(res['1']), 'b' + marker)
    ax.plot(res['mdl'], np.array(res['ttl']) ,  'r'+ marker )
#     ax.plot(res['mdl'], np.array(res['mask']) * res['paras']['m'] + np.array(res['nomask']) * (1 - res['paras']['m']), 'r' + marker)
    append_legend_list(legend_list, res['mdl'], 'analysis')


def plot_sim(res_list, paras, mdl, ax, legend_list, marker='+', itemname='pe'):
    ax.plot(mdl, np.array(res_list[0][itemname]), 'g' + marker)
    ax.plot(mdl, np.array(res_list[1][itemname]), 'b' + marker)
    ax.plot(mdl, np.array(res_list[0][itemname]) * paras['m'] + np.array(res_list[1][itemname]) * (1 - paras['m']), 'r' + marker)
    append_legend_list(legend_list, mdl, 'sim')
    
def scatter_sim(res_list, ax, paras, mdl, legend_list, marker='o', ):
    ax.scatter(mdl, np.array(res_list[0]['pe']), marker=marker, facecolors='none', edgecolors='g')
    ax.scatter(mdl, np.array(res_list[1]['pe']), marker=marker, facecolors='none', edgecolors='b')
    ax.scatter(mdl, np.array(res_list[0]['pe']) * paras['m'] + np.array(res_list[1]['pe']) * (1 - paras['m']), 'r' + marker)
    append_legend_list(legend_list, mdl, 'sim')
    
def set_ax(legend_list, x_label, y_label, title, ax):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(legend_list)
    ax.set_title(title) 
    
    
### For figure 7 to 10 ###
def ax_plot(ax, th_list, sim_list, title, xlabel, ylabel, x_axis):    
    assert len(th_list) == len(sim_list)
    n_lines = len(th_list)
    assert n_lines <= 4
    th_line_styles = ['g-.', 'b--', 'y-', 'r:'] # only allow 3 types of masks and 1 ttl
    sim_line_styles = ['g^', 'bd', 'yo', 'rs']


    for idx, linei in enumerate(range(n_lines - 1)):
        ax.plot(x_axis, th_list[idx], th_line_styles[idx])
    ax.plot(x_axis, th_list[-1], th_line_styles[-1])

    for idx, linei in enumerate(range(n_lines - 1)):
        ax.plot(x_axis, sim_list[idx], sim_line_styles[idx], fillstyle='none')
    ax.plot(x_axis, sim_list[-1], sim_line_styles[-1], fillstyle='none')

    if title != None:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def get_setting_txt(which_figure, m2, tm1, tm2, T, n_nodes, n_exps, md, thr):
    if (which_figure == 8) or (which_figure == 10):
        setting_txt =  "m0 + m1 + m2 = 1, m2=%.1f, Mean degree = %d, tm1 = [%.1f, %.1f, 1], tm2 = [%.1f, %.1f, 1], T = %.1f, %dn_e%d, thr=%.6f"%( 
                            m2,        
                            md,
                            tm1[0],
                           tm1[1],
                           tm2[0],
                           tm2[1],
                           T,
                           n_nodes, 
                           n_exps,
        thr) 
        
    if which_figure == 7:
        setting_txt =  "m0 + m1 = 1, Mean degree = %d, tm1 = [%.1f, %.1f], tm2 = [%.1f, %.1f], T = %.1f, %dn_e%d, thr=%.6f"%(         
                            md,
                            tm1[0],
                           tm1[1],
                           tm2[0],
                           tm2[1],
                           T,
                           n_nodes, 
                           n_exps, thr) 
    return setting_txt


def plot_f8(data, setting_txt, tm1, tm2, m2, n_nodes, n_exps, T, md, thr):
    
    m0_list = data['m0_list']
    m1_list = data['m1_list']
    es_th_list = data['es_th_list']
    es_sim_list = data['es_sim_list']
    pe_th_list = data['pe_th_list']
    pe_sim_list = data['pe_sim_list']
    
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharex=True,
                                        figsize=(12, 5))
    fig.tight_layout(pad=4)
    
    ax0.set_ylim(0, 1)
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    
        
    ## ax0: PE ##     
    ax_plot(ax0, 
        pe_th_list, pe_sim_list,
        "Probability of Emergence\n", 
        r'$m_{surgical}$',  
        "Probability", m0_list)
    
    ## ax1: ES(Given Emergence) ##
    ax_plot(ax1, es_th_list, es_sim_list, 
            "Epidemic Size (Given Emergence)\n", 
            r'$m_{surgical}$',  
            "Fraction", m0_list)
    
    ## ax2: Individual ES ##   
    individual_es_th_list = [es_th_list[0] / np.array(m0_list), 
                             es_th_list[1] / np.array(m1_list),
                             es_th_list[2] / m2,
                             es_th_list[3]]

    individual_es_sim_list = [es_sim_list[0] / np.array(m0_list), 
                              es_sim_list[1] / np.array(m1_list), 
                              es_sim_list[2] / m2, 
                              es_sim_list[3]]


    ax_plot(ax2, 
            individual_es_th_list, 
            individual_es_sim_list, 
            "Individual Infection Probability\n", 
            r'$m_{surgical}$',  
            'Fraction', 
            m0_list)

    ax2.legend(['Theory Surgical',
        'Theory Cloth',
        'Theory No mask',
        'Theory Random',

        'Sim Surgical',
        'Sim Cloth',
        'Sim No mask',
        'Sim Total(Random)',], 
        bbox_to_anchor=(1.05, 1.0), loc='upper left', )
    print(setting_txt)
    plt.savefig('high_res_figures/varying_m0_md%d_m2_%.1f_%dn_e%d_%.3f.eps'%(md, m2, n_nodes, 
                           n_exps,
                            thr), format='eps', bbox_inches='tight')
    
    
def plot_f10(data, setting_txt, tm1, tm2, m2, n_nodes, n_exps, T, md, thr):
    m0_list = data['m0_list']
    m1_list = data['m1_list']
    es_th_list = data['es_th_list']
    es_sim_list = data['es_sim_list']
    pe_th_list = data['pe_th_list']
    pe_sim_list = data['pe_sim_list']
    
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharex=True,
                                        figsize=(12, 5))
    fig.tight_layout(pad=4)
        
    ## ax0 Prob ##     
    ax_plot(ax0, 
            pe_th_list, pe_sim_list,
            "Probability of Emergence\n", 
            r'$m_{outward-good}$', 
            "Probability", m0_list)

    
    ## ax1 ##   
    ax_plot(ax1, es_th_list, es_sim_list, 
            "Epidemic Size (Given Emergence)\n", 
            r'$m_{outward-good}$', 
            "Fraction", m0_list)
    
    

    ## ax2 ##
    exs_th_list = [es_th_list[0] * pe_th_list[0], 
                  es_th_list[1] * pe_th_list[1],
                  es_th_list[2] * pe_th_list[2],
                  es_th_list[3] * pe_th_list[3],]

    exs_sim_list = [es_sim_list[0] * pe_sim_list[0], 
                    es_sim_list[1] * pe_sim_list[1],
                    es_sim_list[2] * pe_sim_list[2],
                    es_sim_list[3] * pe_sim_list[3],]
    ax_plot(ax2, 
        exs_th_list, exs_sim_list,
        "Expected Fraction of Infection\n", 
        r'$m_{outward-good}$', 
        "Fraction", m0_list)
    


    ax2.legend(['Theory Outward-good',
                'Theory Inward-good',
                'Theory No mask',
                'Theory Total(Random)',

                'Sim Outward-good',
                'Sim Inward-good',
                'Sim No mask',
                'Sim Total(Random)',], bbox_to_anchor=(1.05, 1), loc='upper left')

    print(setting_txt)
    plt.savefig('high_res_figures/varying_mout_md%d_m2_%.1f_%dn_e%d_%.3f.eps'%(md, m2, n_nodes, 
                           n_exps,
                            thr), format='eps', bbox_inches='tight')

def plot_f7(data, setting_txt, tm1, tm2, m2, n_nodes, n_exps, T, md, thr):
    m0_list = data['m0_list']
    m1_list = data['m1_list']
    es_th_list = data['es_th_list']
    es_sim_list = data['es_sim_list']
    pe_th_list = data['pe_th_list']
    pe_sim_list = data['pe_sim_list']
    
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharex=True,
                                        figsize=(12, 5))
    fig.tight_layout(pad=4)
    
        
    ## ax0: PE ##     
    ax_plot(ax0, 
        pe_th_list, pe_sim_list,
        "Probability of Emergence\n", 
        r'$m_{surgical}$',  
        "Probability", m0_list)
    
    ## ax1: ES(Given Emergence) ##
    ax_plot(ax1, es_th_list, es_sim_list, 
            "Epidemic Size (Given Emergence)\n", 
            r'$m_{surgical}$',  
            "Fraction", m0_list)
    
    ## ax2: Individual ES ##   
    individual_es_th_list = [es_th_list[0] / np.array(m0_list), 
                             es_th_list[1] / np.array(m1_list),
                             es_th_list[2]]

    individual_es_sim_list = [es_sim_list[0] / np.array(m0_list), 
                              es_sim_list[1] / np.array(m1_list), 
                              es_sim_list[2]]


    ax_plot(ax2, 
            individual_es_th_list, 
            individual_es_sim_list, 
            "Individual Infection Probability\n", 
            r'$m_{surgical}$',  
            'Fraction', 
            m0_list)

    ax2.legend(['Theory Surgical',
        'Theory Cloth',
#         'Theory No mask',
        'Theory Total(Random)',

        'Sim Surgical',
        'Sim Cloth',
#         'Sim No mask',
        'Sim Total(Random)',], 
        bbox_to_anchor=(1.05, 1.0), loc='upper left', )
    print(setting_txt)
    plt.savefig('high_res_figures/varying_m0_md%d_%dn_e%d_%.3f.eps'%(md, n_nodes, 
                           n_exps,
                            thr), format='eps', bbox_inches='tight')
    
def figure7_to10(tm1, tm2, m2, n_nodes, n_exps, T, md, which_figure, thr, cp):
    
    change_m0_data = load_change_m0_data(tm1, tm2, m2, n_nodes, n_exps, T, md, which_figure, thr, cp)
    
    setting_txt = get_setting_txt(which_figure, m2, tm1, tm2, T, n_nodes, n_exps, md, thr)

    if which_figure == 7:
        plot_f7(change_m0_data, setting_txt, tm1, tm2, m2, n_nodes, n_exps, T, md, thr)
        
    if which_figure == 8:
        plot_f8(change_m0_data, setting_txt, tm1, tm2, m2, n_nodes, n_exps, T, md, thr)
        
    if which_figure == 10:
        plot_f10(change_m0_data, setting_txt, tm1, tm2, m2, n_nodes, n_exps, T, md, thr)
    
    return change_m0_data
        
def get_processed_res(thr=0.05, 
                      n=5000, 
                      e=1000, 
                      m=0.45, 
                      T=0.6, 
                      tm1=0.3, 
                      tm2=0.7, 
                      kmax=20,
                      maxd=10,
                      mind=10,
                      ns=1,
                      alpha=0,
                      mdl1=None,
                      mdl2=None,
                      checkpoint=100, 
                      msg='0to3', 
                      time_exp='', 
                      modelname='', 
                      change_metric='m'):
    
    res = load_sim_raw_results(m=m, 
                               T=T, 
                               tm1=tm1, 
                               tm2=tm2, 
                               kmax=kmax,
                               maxd=maxd,
                               mind=mind,
                               ns=ns,
                               alpha=alpha,
                               mdl1=mdl1,
                               mdl2=mdl2,
                               msg=msg,
                               modelname=modelname,
                               change_metric=change_metric,
                               n=n,
                               e=e,
                               checkpoint=checkpoint,
                               time_exp=time_exp,)
    
    res_list = process_raw(res['raw'], res['paras'], thr)
    
    return res_list, res['paras'], res['mdl']



def load_analysis(item, tm1, tm2, m2, T, md, m0_list, m1_list, which_figure,):
    list_0 = []
    list_1 = []
    list_2 = []
    list_ttl = []
    
    for m0idx, m0 in enumerate(m0_list):
        m1 = m1_list[m0idx]
        if m2 == 0:
            m = [m0, m1,]
        else:
            m = [m0, m1, m2]
        
        msg = get_msg(item, m0, m1, m2, md, which_figure)

        analysis = load_analysis_results(m=m, 
                                    T=T, 
                                    tm1=tm1, 
                                    tm2=tm2, 
                                    itemname=item, 
                                    msg=msg,)

        list_0.append(analysis['0'][0])
        list_1.append(analysis['1'][0])
        if which_figure != 7:
            list_2.append(analysis['2'][0])
        list_ttl.append(analysis['ttl'][0])


    if item == 'es':
        theory_0 = np.array(list_0) * (np.array(m0_list))
        theory_1 = np.array(list_1) * (np.array(m1_list))
        theory_2 = np.array(list_2) * m2
        theory_ttl = np.array(list_ttl)
    if item == 'pe':
        theory_0 = np.array(list_0) 
        theory_1 = np.array(list_1) 
        theory_2 = np.array(list_2) 
        theory_ttl = np.array(list_ttl)    
 
    
    if which_figure == 7:
        th_list = [theory_0, theory_1, theory_ttl]
    else:
        th_list = [theory_0, theory_1, theory_2, theory_ttl]
    
    return th_list

def get_msg(item, m0, m1, m2, md, which_figure):
    # For simulation, item == 'es'
    if which_figure == 7:
        msg = item + '_check_m_impact_md%d_m2_%.1f_%.1f'%(md, m2, m0)

    elif which_figure == 8:
        msg = item + '_check_m_impact_md10_m2_%.1f_%.1f'%(m2, m0)

    elif which_figure == 10:
        msg = item + '_check_inout_0.1_0.9_impact_md10_m2_%.1f_%.1f'%(m2, m0)
#         msg = item + '_check_inout_0.2_0.8_impact_md10_m2_%.1f_%.1f'%(m2, m0)
    elif which_figure == -1: # new test
        msg = 'new_%s_%d_%.1f'%(item, md, m0)
    else:
        msg = 'None'
    return msg

def load_sim(item, n_nodes, n_exps, cp, tm1, tm2, m2, T, md, m0_list, m1_list, thr, which_figure,):
    sim0 = []
    sim1 = []
    sim2 = []
    sim = []
    
    for m0idx, m0 in enumerate(m0_list):
        m1 = m1_list[m0idx]
        if m2 == 0:
            m = [m0, m1,]
        else:
            m = [m0, m1, m2]
        
        msg = get_msg('es', m0, m1, m2, md, which_figure) # For sim, es and pe both use 'es'

        sims, paras, mdls = get_processed_res(thr=thr, n=n_nodes, e=n_exps, 
                             m=m, T=T, 
                             tm1=tm1, tm2=tm2, 
                             checkpoint=cp, 
                             msg=msg, modelname='mask',)

    
        if item == 'es':
            sim0.append(sims[0]['es0'][0])
            sim1.append(sims[0]['es1'][0])
            if which_figure != 7:
                sim2.append(sims[0]['es2'][0])
            sim.append(sims[0]['es'][0])
        if item == 'pe':
            sim0.append(sims[0]['pe'][0])
            sim1.append(sims[1]['pe'][0])
            if which_figure != 7:
                sim2.append(sims[2]['pe'][0])
 
    if item == 'es':
        if which_figure == 7:
            sim_list = [sim0, sim1, sim]
        else:
            sim_list = [sim0, sim1, np.array(sim2), sim]
    if item == 'pe':
        if which_figure == 7:
            sim = np.array(sim0) * np.array(m0_list) + np.array(sim1) * np.array(m1_list)
            sim_list = [np.array(sim0),  np.array(sim1), sim]
        else:
            sim = np.array(sim0) * np.array(m0_list) + np.array(sim1) * np.array(m1_list) + np.array(sim2) * m2
            sim_list = [np.array(sim0),  np.array(sim1), np.array(sim2), sim]
    return sim_list
    
def get_m01_list(m2):
    if m2 == 0:
        m0_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
    if m2 == 0.1:
        m0_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    if m2 == 0.2:
        m0_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,]

    if m2 == 0.4:
        m0_list = [0.1, 0.2, 0.3, 0.4, 0.5, ]


    m1_list = m0_list.copy()
    m1_list.sort(reverse = True)
    return m0_list, m1_list
    
def load_change_m0_data(tm1, tm2, m2, n_nodes, n_exps, T, md, which_figure, thr, cp):
    change_m0_data = dict()
    m0_list, m1_list = get_m01_list(m2)    
    change_m0_data['m0_list'] = m0_list
    change_m0_data['m1_list'] = m1_list
    change_m0_data['es_th_list'] = load_analysis('es', tm1, tm2, m2, T, md, m0_list, m1_list, which_figure,)
    change_m0_data['pe_th_list'] = load_analysis('pe', tm1, tm2, m2, T, md, m0_list, m1_list, which_figure,)
    change_m0_data['es_sim_list'] = load_sim('es', n_nodes, n_exps, cp, tm1, tm2, m2, T, md, m0_list, m1_list, thr, which_figure,)
    change_m0_data['pe_sim_list'] = load_sim('pe', n_nodes, n_exps, cp, tm1, tm2, m2, T, md, m0_list, m1_list, thr, which_figure,)

    return change_m0_data

def get_indiv_exp(m, T, tm1, tm2, kmax, maxd, mind, ns, alpha, mdl1, mdl2, msg, modelname, change_metric, n, e, cp_size, time_exp):
    '''
    Funciton: Get each indiviudal trial result
    Return: list(list), len(outside list) = number of mask types, len(inside list) == total number of experiments
    '''
    res = load_sim_raw_results(m=m, 
           T=T, 
           tm1=tm1, 
           tm2=tm2, 
           kmax=kmax,
           maxd=maxd,
           mind=mind,
           ns=ns,
           alpha=alpha,
           mdl1=None,
           mdl2=None,
           msg=msg,
           modelname=modelname,
           change_metric=change_metric,
           n=n,
           e=e,
           checkpoint=cp_size,
           time_exp=time_exp,)
    
    n_cp = e/cp_size
    n_mask_types = len(tm1)
    md = tuple(maxd) # assume that maxd == mind == (md1, md2)
    all_res = []
    for ss in range(n_mask_types):
        ss_res = []
        for cpi in range(int(n_cp)):
            cpi_res = res['raw'][ss][md][cpi+1]
            for i in range(int(cp_size)):
                ss_res.append(cpi_res[i][0])
        all_res.append(ss_res)
    return all_res

def save_scatter_indiv(raw_res, ss, n, e, maxd):
    # Separate ss0, ss1, ...
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter([maxd[0]] * len(raw_res[ss]), raw_res[ss], alpha=0.05)
    plt.title(str(maxd) + '\nss%d'%(ss))
    plt.savefig('indiv_exp30/exp30_%dn_%de_ss%d_md[%.2f_%.2f].png'%(n, e, ss, maxd[0], maxd[1]))
    
    # Combine ss
    combined_res = []
    for ss in range(len(raw_res)):
        combined_res.extend(raw_res[ss])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter([maxd[0]] * len(combined_res), combined_res, alpha=0.05)
    plt.savefig('indiv_exp30/exp30_%dn_%de_md[%.2f_%.2f].png'%(n, e, maxd[0], maxd[1]))