# %%
# Importing the data

# %%
import numpy as np
from scipy.integrate import simpson
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

def Arr(T, dH, T_ref):
    in_exp = dH/8.3145*(1/T - 1/T_ref)
    thet = np.exp(in_exp)
    return thet


def SpreP(iso_pure, P, T, N_pi = 41):
    if P <= 1E-9 and P >= 0:
        return 0
    P_dom = np.linspace(0, P, N_pi,)
    q_dom = iso_pure(P_dom, T)
    P_dom[0] = 1E-8
    within_integ = q_dom/P_dom
    pi_ov_RT = simpson(within_integ, P_dom)
    return pi_ov_RT

sigmoid = lambda x: 1/(1+np.exp(-x))
logit = lambda x: np.log(x/ (1-x))

def err_SpreP(x_list, iso_list, P_list, T):
    is_binary = False
    for ii in range(len(P_list)):
        if P_list[ii] < 1E-5:
            is_binary = True
            P_list[ii] = 0
    P_ov = np.sum(P_list)
    y_list = np.array(P_list)/P_ov
    P_vap_list = []
    pi_list = []
    for pp,xx,iso in zip(P_list ,x_list, iso_list):
        P_vap_tmp = pp/(xx+1E-7)
        pi_tmp = SpreP(iso, P_vap_tmp, T)

        P_vap_list.append(P_vap_tmp)
        pi_list.append(pi_tmp)
    err_sum = 0
    for ii in range(len(pi_list)-1):
        err_sum += (pi_list[ii] - pi_list[ii+1])**2
    err_sum += (pi_list[-1] - pi_list[0])**2
    return err_sum
def obj_fun(ligit_x_no_end, iso_list, P_list, T):
    x_list = []
    for ligitx in ligit_x_no_end:
        x_list.append(sigmoid(ligitx)) # Use of sigmoid f'n

    x_end = 1-np.sum(x_list) # sum(mole frac) = 1
    x_list.append(x_end) # 3rd component
    constr = 0
    if x_end < 0:
        constr += 5E5*x_end**2
    err_pi = err_SpreP(x_list, iso_list, P_list, T)
    err_const_sum = err_pi + constr
    return err_const_sum

#def obj_fun_bi(ligit_x_no_end, iso_list, P_list, T):
#    x_list = []

# %%
# Final product function !
# %%
def IAST_tern(P_arr, T, iso_list):
    ligit_x_init = [-0.2,]*(len(iso_list)-1)
    opt_res1 = differential_evolution(obj_fun,
                                      [[-6,6],]*len(ligit_x_init),
                                      args=(iso_list ,P_arr, T),
                                      maxiter = 1500)
    opt_res2 = minimize(obj_fun, opt_res1.x,
                        args=(iso_list, P_arr, T,),
                        method = 'BFGS')
    x_res = sigmoid(opt_res2.x)
    x_end = 1-np.sum(x_res)
    x_list = list(x_res) + [x_end]
    P_vap_list = np.array(P_arr)/(np.array(x_list) + 1E-7)
    q_not_list = []
    x_ov_q_list = []
    for iso, pp, xx in zip(iso_list, P_vap_list, x_list):
        q_tmp = iso(pp, T)
        x_ov_q_tmp = xx/q_tmp
        q_not_list.append(q_tmp)
        x_ov_q_list.append(x_ov_q_tmp)
    q_tot = 1/np.sum(x_ov_q_list)
    q_arr = np.array(x_list)*q_tot
    return q_arr



# %%
# TESTING the FUNCTIONS

# %%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # =-=-=-=-=-=-=-=-=-=-=-=-= #
    # =- Isotherm example 01 -= #
    # =-=-=-=-=-=-=-=-=-=-=-=-= #
    qm1 = 4.9   # mol/kg
    K1 = 0.02   # bar^-1
    K2 = 0.04   # bar^-2
    dH1 = 12000 # J/mol
    T_ref1 = 300
    def iso1(P, T): # Quadratic
        P_norm = P*Arr(T, dH1, T_ref1)
        numer = qm1*(K1*P_norm + 2*K2*P_norm**2)
        denom = 1 + K1*P_norm + K2*P_norm**2
        q = numer/denom
        return q
    # testiso1
    P_dom = np.linspace(0,20,61)
    q1_res = iso1(P_dom, 300)
    #plt.plot(P_dom, q1_res)
    #plt.show()

    # =-=-=-=-=-=-=-=-=-=-=-=-= #
    # =- Isotherm example 02 -= #
    # =-=-=-=-=-=-=-=-=-=-=-=-= #
    # Langmuir
    qm2 = 3.2   # mol/kg @ 300 K
    b2 = 0.9   # bar^-1  @ 300 K
    dH2 = 9500 # J/mol
    T_ref2 = 300

    def iso2(P, T): # Langmuir
        P_norm = P*Arr(T, dH2, T_ref2)
        q = qm2*b2*P_norm/(1+b2*P_norm)
        return q
    P_dom = np.linspace(0,20,61)
    q2_res = iso2(P_dom, 300)
    #plt.plot(P_dom, q2_res)
    #plt.show()

    # =-=-=-=-=-=-=-=-=-=-=-=-= #
    # =- Isotherm example 03 -= #
    # =-=-=-=-=-=-=-=-=-=-=-=-= #
    # Dual-site
    qm3_1 = 0.2   # mol/kg @ 300 K
    qm3_2 = 1.2   # mol/kg @ 300 K
    b3_1 = 2.2   # bar^-1  @ 300 K
    b3_2 = 0.1   # bar^-1  @ 300 K

    dH3 = 8800 # J/mol
    T_ref3 = 300

    def iso3(P, T): # Dual-site Langmuir
        P_norm = P*Arr(T, dH3, T_ref3)
        q = qm3_1*b3_1*P_norm/(1+b3_1*P_norm) + qm3_2*b3_2*P_norm/(1+b3_2*P_norm)
        return q
    P_dom = np.linspace(0,20,61)
    q3_res = iso3(P_dom, 300)
    #plt.plot(P_dom, q3_res)
    #plt.show()
    qm4 = 1.1
    b4 = 0.05
    n4 = 0.3
    dH4 = 4000
    T_ref4 = 300
    def iso4(P,T):
        P_norm = P*Arr(T,dH4, T_ref4)
        q = qm4*b4*P_norm/(1+b4*P_norm)
        return q

    pi1 = SpreP(iso1, 5, 300)
    print(pi1)
    P_dom = np.linspace(0, 5, 100)
    pi1_list = []
    for pp in P_dom:
        pi1_list.append(SpreP(iso1, pp, 300))
    '''
    plt.plot(P_dom, iso1(P_dom, 300),
             label = 'isotherm')
    plt.plot(P_dom, pi1_list,
             label = 'spreading P ')
    plt.legend()
    plt.show()
    '''
    iso_list_test = [iso1, iso2, iso3]
    P_list_test = [0.5,1.2,0.3,]
    T_test = 300
    x3_list = [0.01, 0.03, 0.05, 0.07]
    x1_dom = np.linspace(0.07, 0.12, 51)
    '''
    for x3 in x3_list[::-1]:
        err_sp_res_list = []
        x2_dom = 1-x1_dom-x3
        for xx1,xx2 in zip(x1_dom, x2_dom):
            err_sum_tmp = err_SpreP([xx1,xx2,x3],
                                    iso_list_test, P_list_test,
                                    T_test)
            err_sp_res_list.append(err_sum_tmp)
        plt.plot(x1_dom, err_sp_res_list,
                 label = f'x3 = {x3}')
    plt.legend(fontsize = 14)
    plt.ylabel('pi err sum ($\sum (\pi_{i} - \pi_{i+1})^{2}$')
    plt.xlabel('mole frac. of comp. 1 (mol/mol) ')
    plt.show()
    '''
    iso_list_test = [iso1,iso2,iso3,]
    iast_res = IAST_tern([0.15, 0.1, 1.75], 300, 
                         iso_list_test,)
    print(iast_res)
    #P1_dom = np.linspace(0.001, 0.5, 51)
    #P2 = 0.1
    #P3_dom = 2-P2-P1_dom
    P_ov = np.linspace(0.02, 5, 26)
    P1_dom = P_ov*0.3
    P2_dom = P_ov*0.2
    P3_dom = P_ov*0.5
    T_test = 300

    q1_list = []
    q2_list = []
    q3_list = []
    '''
    for P1,P2, P3 in zip(P1_dom, P2_dom, P3_dom):
        q1,q2,q3 = IAST_tern([P1,P2,P3,], T_test,
                                 iso_list_test)
        q1_list.append(q1)
        q2_list.append(q2)
        q3_list.append(q3)
    x_axi = P_ov
    plt.plot(x_axi, q1_list, label='q1')
    plt.plot(x_axi, q2_list, label='q2')
    plt.plot(x_axi, q3_list, label='q3')
    plt.xlabel('P1 (bar)')
    plt.ylabel('uptake (mol/kg)')
    plt.legend(fontsize = 13)
    plt.show()
    '''
    iso_list_test2 = [iso1, iso2]
    q1,q2 = IAST_tern([0.2, 0.5], 
                      T_test, iso_list_test2)
    
    # Quadraple
    iso_list_test4 = [iso1,iso2,iso3,iso4]
    q1234 = IAST_tern([0.3, 0.4, 1, 1],
                            T_test, iso_list_test4)
    print(q1234)

        









print('END')