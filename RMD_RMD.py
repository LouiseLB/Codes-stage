import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sp
from scipy.integrate import solve_ivp


# PARAMETRES POUR RMD

# Capacitance 
C_rmd = 1.2

# Potentiel d'equilibre
E_K = -80
E_Ca = 60
E_Na = 30
E_L = -80

# Conductances
g_SHL1 = 2.5 #2.5
g_SHK1 = 1.1 #1.1
g_IRK = 0.2 #0.2
g_EGL36 = 1.3 #1.3
g_EGL19 = 0.99 #0.99
g_UNC2 = 0.9 #0.9
g_CCA1_1 = 3.1 #3.1
g_NCa = 0.05 #0.05
g_L_rmd_1 = 0.4 #0.4

###
# Tous les courants des modèles
###

#NEURONE RMD

#CANAUX POTASSIQUES VOLTAGE DEPENDANTS
# SHL1

def m_SHL1inf(V):
    return 1/(1+np.exp((-V-6.8)/(14.1)))

def h_SHL1inf(V):
    return 1/(1+np.exp((V+51.1)/8.3))  #31.1 à la place de 51.1

def tau_mSHL1(V):
    return 1.4/(np.exp((-17.5-V)/12.9)+np.exp((V+3.7)/6.5)) + 0.2

def tau_hSHL1f(V):
    return 53.9/(1+np.exp((V+28.2)/4.9)) + 2.7

def tau_hSHL1s(V):
    return 842.2/(1+np.exp((V+37.7)/6.4)) + 11.9

def I_SHL1(m,hf,hs,V):
    return g_SHL1*np.power(m,3)*(0.7*hf+0.3*hs)*(V-E_K)


#SHK1

def m_SHK1inf(V):
    return 1/(1+np.exp((20.4-V)/7.7))

def h_SHK1inf(V):
    return 1/(1+np.exp((V+7)/5.8))

def tau_mSHK1(V):
    return 26.6/(np.exp((-33.7-V)/15.4)+np.exp((V+33.7)/15.8)) + 2.0 

tau_hSHK1 = 1400

def I_SHK1(m,h,V):
    return g_SHK1*m*h*(V-E_K)


#EGL36

def m_EGL36inf(V):
    return 1/(1+np.exp((63-V)/28.5))

tau_mEGL36s = 355
tau_mEGL36m = 63
tau_mEGL36f = 13

def I_EGL36(mf,mm,ms,V):
    return g_EGL36*(0.33*mf+0.36*mm+0.39*ms)*(V-E_K)


#IRK

def m_IRKinf(V):
    return 1/(1+np.exp((V+82)/13))

def tau_mIRK(V):
    return 17.1/(np.exp((-17.8-V)/20.3)+np.exp((V+43.4)/11.2)) + 3.8

def I_IRK(m,V):
    return g_IRK*m*(V-E_K)
 

#COURANTS CALCIQUES VOLTAGE DEPENDANTS

#EGL19

def m_EGL19inf(V):
    return 1/(1+np.exp((-4.4-V)/7.5))

def h_EGL19inf(V):
    return ((1.43/(1+np.exp((14.9-V)/12))) + 0.14)*((5.96/(1+np.exp((V+20.5)/8.1))) + 0.6)

def tau_mEGL19(V):
    return 2.9*np.exp(-np.power(((V+4.8)/6),2)) + 1.9*np.exp(-np.power(((V+8.6)/30),2)) + 2.3

def tau_hEGL19(V):
    return 0.4*(44.6/(1+np.exp((V+33)/5)) + 36.4/(1+np.exp((V-18.7)/3.7)) + 43.1)

def I_EGL19(m,h,V):
    return g_EGL19*m*h*(V-E_Ca)

#UNC2

def m_UNC2inf(V):
    return 1/(1+np.exp((-37.2-V)/4))

def h_UNC2inf(V):
    return 1/(1+np.exp((V+77.5)/5.6))

def tau_mUNC2(V):
    return 1.5/(np.exp((-38.2-V)/9.1)+np.exp((V+38.2)/15.4)) + 0.3 #1.5 à la place de 4.47 et 0.1 à la place de 0.3

def tau_hUNC2(V):
    return 142.5/(1+np.exp((V-22.9)/3.5)) + 122.6/(1+np.exp((-7-V)/3.6))

def I_UNC2(m,h,V):
    return g_UNC2*m*h*(V-E_Ca)

#CCA1

def m_CCA1inf(V):
    return 1/(1+np.exp((-57.7-V)/2.4))

def h_CCA1inf(V):
    return 1/(1+np.exp((V+73)/8.1))

def tau_mCCA1(V):
    return 20/(1+np.exp((-92.5-V)/-21.1)) + 0.4

def tau_hCCA1(V):
    return  22.4/(1+np.exp((V+75.7)/9.4)) + 1.6

def I_CCA1(g_CCA1,m,h,V):
    return g_CCA1*m*m*h*(V-E_Ca)


#COURANTS DE FUITE

#NCA

def I_NCA(V):
    return g_NCa*(V-E_Na)

#LEAK

def I_L(g_L_rmd,V):
    return g_L_rmd*(V-E_L)



tau=0.042

y0_bis=-40
E21=0


#CONDUCTANCE SYNAPTIQUE

def g_inf(V):
    return 0.4/(1+np.exp((0.5-V)/(10)))   

# COURANT SYNAPTIQUE

def I_c(V1,V2):
    return g_inf(V1)*(V2-E21)


def Istim_rmd(I,t):
    if (t>=200 and t<=800):
        return I
    else: 
        return 0

g=np.vectorize(Istim_rmd) 

def I_inf(V):
    return (I_SHL1(m_SHL1inf(V),h_SHL1inf(V),h_SHL1inf(V),V) + I_NCA(V) + I_L(0.4,V) +I_SHK1(m_SHK1inf(V),h_SHK1inf(V),V)+I_IRK(m_IRKinf(V),V)+I_EGL36(m_EGL36inf(V),m_EGL36inf(V),m_EGL36inf(V),V)+I_EGL19(m_EGL19inf(V),h_EGL19inf(V),V)+I_UNC2(m_UNC2inf(V),h_UNC2inf(V),V)+I_CCA1(3.1,m_CCA1inf(V),h_CCA1inf(V),V))

def f(y, t, I):
    dy = np.zeros((32,))

    V_rmd_1 = y[0]
    mSHL1_1 = y[1]
    hSHL1f_1 = y[2]
    hSHL1s_1 = y[3]
    mSHK1_1 = y[4]
    hSHK1_1 =y[5]
    mIRK_1 = y[6]
    mEGL36f_1 = y[7]
    mEGL36m_1 = y[8]
    mEGL36s_1 = y[9]
    mEGL19_1 = y[10]
    hEGL19_1 = y[11]
    mUNC2_1 = y[12]
    hUNC2_1 = y[13] 
    mCCA1_1 = y[14]
    hCCA1_1 = y[15]
    V_rmd_2 = y[16]
    mSHL1_2 = y[17]
    hSHL1f_2 = y[18]
    hSHL1s_2 = y[19]
    mSHK1_2 = y[20]
    hSHK1_2 =y[21]
    mIRK_2 = y[22]
    mEGL36f_2 = y[23]
    mEGL36m_2 = y[24]
    mEGL36s_2 = y[25]
    mEGL19_2 = y[26]
    hEGL19_2 = y[27]
    mUNC2_2 = y[28]
    hUNC2_2 = y[29] 
    mCCA1_2 = y[30]
    hCCA1_2 = y[31]

    dy[0] = (1/C_rmd)*(g(I,t)-I_SHL1(mSHL1_1,hSHL1f_1,hSHL1s_1,V_rmd_1) - I_NCA(V_rmd_1) - I_L(0.4,V_rmd_1) -I_SHK1(mSHK1_1,hSHK1_1,V_rmd_1)-I_IRK(mIRK_1,V_rmd_1)-I_EGL36(mEGL36f_1,mEGL36m_1,mEGL36s_1,V_rmd_1)-I_EGL19(mEGL19_1,hEGL19_1,V_rmd_1)-I_UNC2(mUNC2_1,hUNC2_1,V_rmd_1)-I_CCA1(3.1,mCCA1_1,hCCA1_1,V_rmd_1)) #dV/dt
    dy[1] = (m_SHL1inf(V_rmd_1)-mSHL1_1)/tau_mSHL1(V_rmd_1) #dmSHL1/dt
    dy[2] = (h_SHL1inf(V_rmd_1)-hSHL1f_1)/tau_hSHL1f(V_rmd_1) #dhSHL1f/dt
    dy[3] = (h_SHL1inf(V_rmd_1)-hSHL1s_1)/tau_hSHL1s(V_rmd_1) #dhSHL1s/dt
    dy[4] = (m_SHK1inf(V_rmd_1)-mSHK1_1)/tau_mSHK1(V_rmd_1) #dmSHK1/dt
    dy[5] = (h_SHK1inf(V_rmd_1)-hSHK1_1)/tau_hSHK1 #dhSHK1/dt
    dy[6] = (m_IRKinf(V_rmd_1)-mIRK_1)/tau_mIRK(V_rmd_1) #dmIRK/dt
    dy[7] = (m_EGL36inf(V_rmd_1)-mEGL36f_1)/tau_mEGL36f #dmEGL36f/dt
    dy[8] = (m_EGL36inf(V_rmd_1)-mEGL36m_1)/tau_mEGL36m #dmEGL36m/dt
    dy[9] = (m_EGL36inf(V_rmd_1)-mEGL36s_1)/tau_mEGL36s #dmEGL36s/dt
    dy[10] = (m_EGL19inf(V_rmd_1)-mEGL19_1)/tau_mEGL19(V_rmd_1) #dmEGL19/dt
    dy[11] = (h_EGL19inf(V_rmd_1)-hEGL19_1)/tau_hEGL19(V_rmd_1) #dhEGL19/dt
    dy[12] = (m_UNC2inf(V_rmd_1)-mUNC2_1)/tau_mUNC2(V_rmd_1) #dmUNC2/dt
    dy[13] = (h_UNC2inf(V_rmd_1)-hUNC2_1)/tau_hUNC2(V_rmd_1) #dhUNC2/dt
    dy[14] = (m_CCA1inf(V_rmd_1)-mCCA1_1)/tau_mCCA1(V_rmd_1) #dmCCA1/dt
    dy[15] = (h_CCA1inf(V_rmd_1)-hCCA1_1)/tau_hCCA1(V_rmd_1) #dhCCA1/dt
    dy[16] = (1/C_rmd)*(-I_SHL1(mSHL1_2,hSHL1f_2,hSHL1s_2,V_rmd_2) - I_NCA(V_rmd_2) - I_L(0.4,V_rmd_2) -I_SHK1(mSHK1_2,hSHK1_2,V_rmd_2)-I_IRK(mIRK_2,V_rmd_2)-I_EGL36(mEGL36f_2,mEGL36m_2,mEGL36s_2,V_rmd_2)-I_EGL19(mEGL19_2,hEGL19_2,V_rmd_2)-I_UNC2(mUNC2_2,hUNC2_2,V_rmd_2)-I_CCA1(3.1,mCCA1_2,hCCA1_2,V_rmd_2)-I_c(V_rmd_1,V_rmd_2)) #dV/dt
    dy[17] = (m_SHL1inf(V_rmd_2)-mSHL1_2)/tau_mSHL1(V_rmd_2) #dmSHL1/dt
    dy[18] = (h_SHL1inf(V_rmd_2)-hSHL1f_2)/tau_hSHL1f(V_rmd_2) #dhSHL1f/dt
    dy[19] = (h_SHL1inf(V_rmd_2)-hSHL1s_2)/tau_hSHL1s(V_rmd_2) #dhSHL1s/dt
    dy[20] = (m_SHK1inf(V_rmd_2)-mSHK1_2)/tau_mSHK1(V_rmd_2) #dmSHK1/dt
    dy[21] = (h_SHK1inf(V_rmd_2)-hSHK1_2)/tau_hSHK1 #dhSHK1/dt
    dy[22] = (m_IRKinf(V_rmd_2)-mIRK_2)/tau_mIRK(V_rmd_2) #dmIRK/dt
    dy[23] = (m_EGL36inf(V_rmd_2)-mEGL36f_2)/tau_mEGL36f #dmEGL36f/dt
    dy[24] = (m_EGL36inf(V_rmd_2)-mEGL36m_2)/tau_mEGL36m #dmEGL36m/dt
    dy[25] = (m_EGL36inf(V_rmd_2)-mEGL36s_2)/tau_mEGL36s #dmEGL36s/dt
    dy[26] = (m_EGL19inf(V_rmd_2)-mEGL19_2)/tau_mEGL19(V_rmd_2) #dmEGL19/dt
    dy[27] = (h_EGL19inf(V_rmd_2)-hEGL19_2)/tau_hEGL19(V_rmd_2) #dhEGL19/dt
    dy[28] = (m_UNC2inf(V_rmd_2)-mUNC2_2)/tau_mUNC2(V_rmd_2) #dmUNC2/dt
    dy[29] = (h_UNC2inf(V_rmd_2)-hUNC2_2)/tau_hUNC2(V_rmd_2) #dhUNC2/dt
    dy[30] = (m_CCA1inf(V_rmd_2)-mCCA1_2)/tau_mCCA1(V_rmd_2) #dmCCA1/dt
    dy[31] = (h_CCA1inf(V_rmd_2)-hCCA1_2)/tau_hCCA1(V_rmd_2) #dhCCA1/dt
    return dy 



V_0_rmd= -70
mSHL1_0 = 0
hSHL1f_0 = 1 # 1
hSHL1s_0 = 1 # 1
mSHK1_0 = 0
hSHK1_0 = 1 #1
mIRK_0 = 0 #0.5 pas écrit sur modelDB
mEGL36f_0 = 0
mEGL36m_0 = 0
mEGL36s_0 = 0
mEGL19_0 = 0
hEGL19_0 = 1 #1
mUNC2_0 = 0
hUNC2_0 = 0.25 #1 0.25
mCCA1_0 = 0
hCCA1_0 = 1 #1

y0 = np.array([V_0_rmd, mSHL1_0, hSHL1f_0, hSHL1s_0, mSHK1_0, hSHK1_0, mIRK_0, mEGL36f_0, mEGL36m_0, mEGL36s_0, mEGL19_0, hEGL19_0, mUNC2_0, hUNC2_0, mCCA1_0, hCCA1_0,V_0_rmd, mSHL1_0, hSHL1f_0, hSHL1s_0, mSHK1_0, hSHK1_0, mIRK_0, mEGL36f_0, mEGL36m_0, mEGL36s_0, mEGL19_0, hEGL19_0, mUNC2_0, hUNC2_0, mCCA1_0, hCCA1_0])

Istim = np.arange(-10, 18, 4)
t = np.arange(0, 1000, 0.5)


for I in Istim: 
        yt2 = odeint(f, y0, t, args=(I,))
        plt.plot(t, yt2[:,0], label=I)       

plt.xlabel(r"$t\;$(ms)",fontsize=20)
plt.ylabel(r"$V(t)\;$(mV)",fontsize=20)
plt.title('Neurone RMD pré',fontsize=35)
plt.tick_params(axis='x', which='major', labelsize=20)
plt.tick_params(axis='y', which='major', labelsize=20)
plt.legend()
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
#plt.savefig(f'RIM_variation_gL+gL_RMD g={g_L_rim}.png')
plt.show()
plt.clf()

for I in Istim:
    yt2 = odeint(f, y0, t, args=(I,))
    plt.plot(t, yt2[:,16], label=I)   
    plt.xlabel(r"$t\;$(ms)",fontsize=20)
plt.ylabel(r"$V(t)\;$(mV)",fontsize=20)
plt.title('Neurone RMD post',fontsize=35)
plt.tick_params(axis='x', which='major', labelsize=20)
plt.tick_params(axis='y', which='major', labelsize=20)
plt.legend()
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
#plt.savefig(f'RIM_variation_gL+gL_RMD g={g_L_rim}.png')
plt.show()
plt.clf()

