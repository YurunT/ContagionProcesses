import numpy as np
from scipy.stats import poisson
from degree_distributions import *
def fi(i, qc, qs, kc, ks, Tc, Ts, m):
    '''
    fi function in the paper derivation for ES
    Note: this function will return the value for a type-i node NOT being infected at level L+1 !!!
    
    Initiator: level 0
    Root node: level infity
    Transmission direction: level 0 -> level infity
    
    Input:
    i : interger ranging from 0 to M-1, representing the mask type for the target node u in the paper
    qc: 1xM np array 
        representing the prob of a node at level L who has a type-c link going to the higher level (L+1) for each type mask
    qs: 1xM np array
        representing the prob of a node at level L who has a type-s link going to the higher level (L+1) for each type mask
    kc: number of type-c links coming from the lower level L 
    ks: number of type-s links coming from the lower level L
    
    (P.S. kc and ks could be excess degree or full degree, but should be considered outside this fun)
    '''
    M = m.shape[0] 
    sum_c = 0
    for j in range(M):
        sum_c += m[j] * (1 - Tc[j][i] + qc[j] * Tc[j][i])
        
    sum_s = 0
    for j in range(M):
        sum_s += m[j] * (1 - Ts[j][i] + qs[j] * Ts[j][i])
        
    return (sum_c ** kc) * (sum_s ** ks)

def solve_q(q_old: np.ndarray, 
            dd_name1: str, dd_paras1: dict, dd_name2: str, dd_paras2: dict, 
            Tc: np.ndarray, Ts: np.ndarray, m: np.ndarray, 
            M: int, alpha: float, kmax: int) -> np.ndarray:
    '''
    This fun will be used by scipy.fsolve to 
    solve for equation (10) and (11) to obtain q_c,infity and q_s,infity
    
    
    Output: 
    [q_c,infity, q_s,infity]
    '''

    # Step 1: obtain the colored degree distribution pd
    # pd: (kmax + 1) x (kmax + 1) matrix. 
    # where pd[i][j] represents the degree prob for a node having colored degree d=[i, j]
    pkc, mm1_c, mm2_c = get_pk(dd_name1, dd_paras1, kmax)
    pks, mm1_s, mm2_s = get_pk(dd_name2, dd_paras2, kmax)
    pd = get_pd(pkc, pks, alpha) 
    md = [0, 0]
    md[0] = mm1_c
    md[1] = mm1_s
    
    
    # Step 2: implement Eq(10)-(11) in ICC submission
    qc_old, qs_old = q_old[:M], q_old[M:]
    
    '''
    Equation (10) in the paper
    '''
    qc_new = np.zeros(M)
    for i in range(M):
        for kc in range(kmax): # full degree of blue edges
            for ks in range(kmax): # full degree of red edges
                qc_new[i] += pd[kc][ks] * kc / md[0] * fi(i, qc_old, qs_old, kc - 1, ks, Tc, Ts, m,)

    '''
    Equation (11) in the paper
    '''
    qs_new = np.zeros(M)
    for i in range(M):
        for kc in range(kmax): # full degree of blue edges
            for ks in range(kmax): # full degree of red edges
                qs_new[i] += pd[kc][ks] * ks / (md[1] * alpha) * fi(i, qc_old, qs_old, kc, ks - 1, Tc, Ts, m,)
    return np.hstack([qc_new, qs_new])

def es_fun_root(q_old: np.ndarray, 
                dd_name1: str, dd_paras1: dict, dd_name2: str, dd_paras2: dict, 
                Tc: np.ndarray, Ts: np.ndarray, m: np.ndarray, 
                M: int, alpha: float, kmax: int):
    '''
    This is the function to be called by scipy.fsolve to obtain the solutions for q_c,infity and q_s,infity
    
    Inputs:
    q_old: 1 x 2M np array, where first M elements represent qc_old, last M elements represent qs_old
    dd_namei: Name of the required degree distribution for the ith layer. Currently support Poisson, PL, PLC.
    dd_parasi: A dictionary of Degree distribution parameters for the ith layer. Poisson(lambda), PL(b), PLC(alpha, kappa).
    Tc: M x M matrix, representing the transmission matrix for type-c links
    Ts: M x M matrix, representing the transmission matrix for type-s links
    m : 1 x M np array, m[i] representing the prob of a type-i mask in the population
    M: total number of mask types in the population
    alpha: prob nodes in C also in S
    kmax: approximation of infity for the purpose of averaging d
    '''
    assert M == m.shape[0] 
    return solve_q(q_old, dd_name1, dd_paras1, dd_name2, dd_paras2, Tc, Ts, m, M, alpha, kmax) - q_old

def get_q(q_infty: np.ndarray, 
            dd_name1: str, dd_paras1: dict, dd_name2: str, dd_paras2: dict, 
            Tc: np.ndarray, Ts: np.ndarray, m: np.ndarray, 
            M: int, alpha: float, kmax: int) -> np.ndarray:

    '''
    This fun is the equation (12) in the paper
    After solving for q_{c,infty} and q_{s,infty}, 
    we can obtain the prob of **NOT** infection for each mask type
    
    Input: 
    q_infty: np 1-d array with length 2M, where first M elements represent qc_infty, last M elements represent qc_infty
    '''
    assert M == m.shape[0] 

    # Step 1: obtain the colored degree distribution pd
    # pd: (kmax + 1) x (kmax + 1) matrix. 
    # where pd[i][j] represents the degree prob for a node having colored degree d=[i, j]
    pkc, mm1_c, mm2_c = get_pk(dd_name1, dd_paras1, kmax)
    pks, mm1_s, mm2_s = get_pk(dd_name2, dd_paras2, kmax)
    pd = get_pd(pkc, pks, alpha) 
    md = [0, 0]
    md[0] = mm1_c
    md[1] = mm1_s
    
    # Step 2: implement Eq(12) in ICC submission
    qc_infty, qs_infty = q_infty[:M], q_infty[M:]
    
    q = np.zeros(M)
    for i in range(M):
        sum_i = 0
        for kc in range(kmax):
            for ks in range(kmax):
                sum_i += pd[kc][ks] * fi(i, qc_infty, qs_infty, kc, ks, Tc, Ts, m)
                
        q[i] = sum_i
    return q