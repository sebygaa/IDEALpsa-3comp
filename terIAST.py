# %%
# Importing the data

# %%
import numpy as np
from scipy.integrate import simpson

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

# %%
# TESTING the FUNCTIONS

# %%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    qm1 = 2.9   # mol/kg
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

    dH3 = 9500 # J/mol
    T_ref3 = 300

    def iso3(P, T): # Langmuir
        P_norm = P*Arr(T, dH3, T_ref3)
        q = qm3_1*b3_1*P_norm/(1+b3_1*P_norm) + qm3_2*b3_2*P_norm/(1+b3_2*P_norm)
        return q
    P_dom = np.linspace(0,20,61)
    q3_res = iso3(P_dom, 300)
    #plt.plot(P_dom, q3_res)
    #plt.show()

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
    x3 = 0.1
    x1_dom = np.linspace(0.1,0.5,100)
    x2_dom = 1-x1_dom-x3
    err_sp_res_list = []
    for xx1,xx2 in zip(x1_dom, x2_dom):
        err_sum_tmp = err_SpreP([xx1,xx2,x3],
                                iso_list_test, P_list_test,
                                T_test)
        err_sp_res_list.append(err_sum_tmp)
    plt.plot(x1_dom, err_sp_res_list)
    plt.ylabel('pi err sum ($\sum (\pi_{i} - \pi_{i+1})^{2}$')
    plt.xlabel('mole frac. of comp. 1 (mol/mol) ')
    plt.show()

    

