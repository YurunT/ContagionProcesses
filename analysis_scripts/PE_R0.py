import math
import sys, site, os, time
import numpy as np
from scipy import optimize 
from scipy.special import comb
from scipy.stats import poisson
import scipy.misc
from datetime import datetime
sys.path.append(os.path.abspath("../auxiliary_scripts/"))
from theory_aux import * 
from input_module import *
from output_module import write_analysis_results
from main_aux import *
from para_settings import para_setting
from numpy import linalg as LA
from degree_distributions import *

def solve_h(h: np.ndarray, 
            dd_name1: str, dd_paras1: dict, dd_name2: str, dd_paras2: dict, 
            Tb: np.ndarray, Tr: np.ndarray, m: np.ndarray, 
            alpha: float, kmax: int):
    
    '''
    h: np 1-d array with length 2M, where first M elements represent hb, last M elements represent hr
    This function implement 2M equations of fb,i and fr,i:
    hb,i = fb,i(hb, hr)
    hr,i = fr,i(hb, hr)
    where, hb = [hb,i], hr = [hr,i], i = 1, ..., M
    2M unknowns, 2M equations
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
    
    
    # Step 2: get new hb and hr
    M = m.shape[0] # num of mask types
    hb, hr = h[:M], h[M:]
    new_hb = np.zeros(M)
    for i in range(M): # hb,i
        hbi = 0
        for j in range(M):
            sum_ = 0
            for kb in range(kmax): # full degree of blue edges
                for kr in range(kmax): # full degree of red edges
                    excess_prob = pd[kb][kr] * kb / md[0]
                    sum_ += excess_prob * (hb[j] ** (kb - 1)) * (hr[j] ** kr)

            hbi += m[j] * (1 - Tb[i][j] + Tb[i][j] * sum_)
        new_hb[i] = hbi


    new_hr = np.zeros(M)
    for i in range(M): # hr,i
        hri = 0
        for j in range(M):
            sum_ = 0
            for kb in range(kmax): # full degree of blue edges
                for kr in range(kmax): # full degree of red edges
                    excess_prob = pd[kb][kr] * kr / (md[1] * alpha) 
                    sum_ += excess_prob * (hr[j] ** (kr - 1)) * (hb[j] ** kb)
            hri += m[j] * (1 - Tr[i][j] + Tr[i][j] * sum_)
        new_hr[i] = hri

    return np.hstack([new_hb, new_hr])


def fun_root(h_old: np.ndarray, 
            dd_name1: str, dd_paras1: dict, dd_name2: str, dd_paras2: dict, 
            Tc: np.ndarray, Ts: np.ndarray, m: np.ndarray, 
            alpha: float, kmax: int):

    '''
    This is the function to be called by scipy.fsolve to obtain the solutions for h_bi and h_ri
    
    Inputs:
    h_old: np 1-d array with length 2M, where first M elements represent hb, last M elements represent hr
    dd_namei: Name of the required degree distribution for the ith layer. Currently support Poisson, PL, PLC.
    dd_parasi: A dictionary of Degree distribution parameters for the ith layer. Poisson(lambda), PL(b), PLC(alpha, kappa).
    Tc: M x M matrix, representing the transmission matrix for type-c links
    Ts: M x M matrix, representing the transmission matrix for type-s links
    m : 1 x M np array, m[i] representing the prob of a type-i mask in the population
    M: total number of mask types in the population
    alpha: prob nodes in C also in S
    kmax: approximation of infity for the purpose of averaging 
    '''
    return solve_h(h_old, dd_name1, dd_paras1, dd_name2, dd_paras2, Tc, Ts, m, alpha, kmax) - h_old


def get_H(h: np.ndarray, dd_name1: str, dd_paras1: dict, dd_name2: str, dd_paras2: dict,
          alpha: float, M: int, kmax: int):
    '''
    h : np array shape of 2 x M
    pd: nd array with the shape (1+kmax, 1+kmax), pd(i,j) equals pkb(kb=i) * pkr(kr=j)
    M: number of mask types
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
    
    # Step 2: get H
    H = np.zeros(M)
    hb, hr = h[:M], h[M:]
    for i in range(M): # Hi
        sum_ = 0
        for kb in range(kmax):
            for kr in range(kmax):
                sum_ += pd[kb, kr] * (hb[i] ** kb) * (hr[i] ** kr) 
        H[i] = sum_       
    return H


def get_spectral_radius(dd_name1: str, dd_paras1: dict, dd_name2: str, dd_paras2: dict,
                        Tb: np.ndarray, Tr: np.ndarray, m: np.ndarray, 
                        alpha: float, kmax: int):
    '''
    Input:

    Note, the Jacobian matrix will be 2M x 2M, 
    where 2 represents 2 graphs, M represents M types of masks
    We need to construct the Jacobian matrix first:
    [[Jbb, Jbr], 
     [Jrb, Jrr]] with shape: 2M x 2M,

    then calculate the spectral radius (largest eigenvalue in absolute value).

    This fun could be used to solve parameters (e.g. Tb, Tr, alpha, etc) for spectral radius = 1 
    '''
    M = m.shape[0] # num of mask types

    # Step 1: obtain the colored degree distribution pd
    # pd: (kmax + 1) x (kmax + 1) matrix. 
    # where pd[i][j] represents the degree prob for a node having colored degree d=[i, j]
    pkc, mm1_c, mm2_c = get_pk(dd_name1, dd_paras1, kmax)
    pks, mm1_s, mm2_s = get_pk(dd_name2, dd_paras2, kmax)
    pd = get_pd(pkc, pks, alpha) 
    md = [0, 0]
    md[0] = mm1_c
    md[1] = mm1_s
    
    # Step 2: construct Jacobian matrix
    # Construct Jbb
    Jbb = np.zeros((M, M))
    for i in range(M): 
        for j in range(M):
            sum_ = 0
            for kb in range(kmax):
                for kr in range(kmax):
                    sum_ += pd[kb, kr] * kb / md[0] * (kb - 1) 
            Jbb[i][j] = m[j] * Tb[i][j] * sum_
    
    # Construct Jrr
    Jrr = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            sum_ = 0
            for kb in range(kmax):
                for kr in range(kmax):
                    sum_ += pd[kb, kr] * kr / (md[1] * alpha) * (kr - 1)
            Jrr[i][j] = m[j] * Tr[i][j] * sum_
            
    # Construct Jbr
    Jbr = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            sum_ = 0
            for kb in range(kmax):
                for kr in range(kmax):
                    sum_ += pd[kb, kr] * kb / (md[0]) * kr
            Jbr[i][j] = m[j] * Tb[i][j] * sum_
            
    # Construct Jrb
    Jrb = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            sum_ = 0
            for kb in range(kmax):
                for kr in range(kmax):
                    sum_ += pd[kb, kr] * kr / (md[1] * alpha) * kb
            Jrb[i][j] = m[j] * Tr[i][j] * sum_
    J = np.hstack([np.vstack([Jbb, Jrb]), np.vstack([Jbr, Jrr])])
    eigvals = LA.eigvals(J)
    s_r = max(abs(eigvals))
    
    return s_r