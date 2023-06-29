import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# PARAMETRES

# Capacitance 
C = 1.2

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
g_CCA1 = 3.1 #3.1
g_NCa = 0.05 #0.05
g_L = 0.4 

Vm= np.arange(-120, 0, 1)

#CANAUX POTASSIQUES VOLTAGE DEPENDANTS
# SHL1

def m_SHL1inf(V):
    return 1/(1+np.exp((-V-6.8)/(14.1)))

def h_SHL1inf(V):
    return 1/(1+np.exp((V+51.1)/8.3)) 

def tau_mSHL1(V):
    return 1.4/(np.exp((-17.5-V)/12.9)+np.exp((V+3.7)/6.5)) + 0.2

def tau_hSHL1f(V):
    return 53.9/(1+np.exp((V+28.2)/4.9)) + 2.7

def tau_hSHL1s(V):
    return 842.2/(1+np.exp((V+37.7)/6.4)) + 11.9

def I_SHL1(m,hf,hs,V):
    return g_SHL1*np.power(m,3)*(0.7*hf+0.3*hs)*(V-E_K)


#Tracer les variables d'activation et d'inactivation
"""
plt.plot(Vm,m_SHL1inf(Vm),'b')  
plt.plot(Vm,h_SHL1inf(Vm),'r')     
plt.show()
"""

#Tracer tau
"""
plt.plot(Vm,tau_mSHL1(Vm),'b')  
plt.plot(Vm,tau_hSHL1f(Vm),'r')   
plt.plot(Vm,tau_hSHL1f(Vm),'g')  
plt.show()
"""


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
    return 1.5/(np.exp((-38.2-V)/9.1)+np.exp((V+38.2)/15.4)) + 0.3 

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

def I_CCA1(m,h,V):
    return g_CCA1*m*m*h*(V-E_Ca)


#COURANTS DE FUITE

#NCA

def I_NCA(V):
    return g_NCa*(V-E_Na)

#LEAK

def I_L(V):
    return g_L*(V-E_L)



# Current injection

def Istim(I,t):
     if (t>=100 and t<=400):
         return I
     elif (t>=650 and t<=850):
         return -15
     else:
         return 0

g = np.vectorize(Istim)


#Steady-state current / Courant d'équilibre
def I_inf(V):
    return (I_SHL1(m_SHL1inf(V),h_SHL1inf(V),h_SHL1inf(V),V) + I_NCA(V) + I_L(V) +I_SHK1(m_SHK1inf(V),h_SHK1inf(V),V)+I_IRK(m_IRKinf(V),V)+I_EGL36(m_EGL36inf(V),m_EGL36inf(V),m_EGL36inf(V),V)+I_EGL19(m_EGL19inf(V),h_EGL19inf(V),V)+I_UNC2(m_UNC2inf(V),h_UNC2inf(V),V)+I_CCA1(m_CCA1inf(V),h_CCA1inf(V),V))


Vm= np.arange(-120, 20, 1)
plt.plot(Vm, I_inf(Vm))
plt.axhline(0, color = "k")
plt.xlabel(r"$V\;$(mV)",fontsize=25)
plt.ylabel(r"$I_{\infty}(V)\;$(pA)",fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=25, length=6, width=2)
plt.tight_layout()
plt.show()


#RESOLUTION DU SYSTEME
t0=0

def f(y, t, I):
    dy = np.zeros((16,))

    V = y[0]
    mSHL1 = y[1]
    hSHL1f = y[2]
    hSHL1s = y[3]
    mSHK1 = y[4]
    hSHK1 =y[5]
    mIRK = y[6]
    mEGL36f = y[7]
    mEGL36m = y[8]
    mEGL36s = y[9]
    mEGL19 = y[10]
    hEGL19 = y[11]
    mUNC2 = y[12]
    hUNC2 = y[13] 
    mCCA1 = y[14]
    hCCA1 = y[15]

    dy[0] = (1/C)*(g(I,t)-I_SHL1(mSHL1,hSHL1f,hSHL1s,V) - I_NCA(V) - I_L(V) -I_SHK1(mSHK1,hSHK1,V)-I_IRK(mIRK,V)-I_EGL36(mEGL36f,mEGL36m,mEGL36s,V)-I_EGL19(mEGL19,hEGL19,V)-I_UNC2(mUNC2,hUNC2,V)-I_CCA1(mCCA1,hCCA1,V)) #dV/dt
    dy[1] = (m_SHL1inf(V)-mSHL1)/tau_mSHL1(V) #dmSHL1/dt
    dy[2] = (h_SHL1inf(V)-hSHL1f)/tau_hSHL1f(V) #dhSHL1f/dt
    dy[3] = (h_SHL1inf(V)-hSHL1s)/tau_hSHL1s(V) #dhSHL1s/dt
    dy[4] = (m_SHK1inf(V)-mSHK1)/tau_mSHK1(V) #dmSHK1/dt
    dy[5] = (h_SHK1inf(V)-hSHK1)/tau_hSHK1 #dhSHK1/dt
    dy[6] = (m_IRKinf(V)-mIRK)/tau_mIRK(V) #dmIRK/dt
    dy[7] = (m_EGL36inf(V)-mEGL36f)/tau_mEGL36f #dmEGL36f/dt
    dy[8] = (m_EGL36inf(V)-mEGL36m)/tau_mEGL36m #dmEGL36m/dt
    dy[9] = (m_EGL36inf(V)-mEGL36s)/tau_mEGL36s #dmEGL36s/dt
    dy[10] = (m_EGL19inf(V)-mEGL19)/tau_mEGL19(V) #dmEGL19/dt
    dy[11] = (h_EGL19inf(V)-hEGL19)/tau_hEGL19(V) #dhEGL19/dt
    dy[12] = (m_UNC2inf(V)-mUNC2)/tau_mUNC2(V) #dmUNC2/dt
    dy[13] = (h_UNC2inf(V)-hUNC2)/tau_hUNC2(V) #dhUNC2/dt
    dy[14] = (m_CCA1inf(V)-mCCA1)/tau_mCCA1(V) #dmCCA1/dt
    dy[15] = (h_CCA1inf(V)-hCCA1)/tau_hCCA1(V) #dhCCA1/dt
    return dy 


#Conditions initiales

V_0= -70
mSHL1_0 = 0
hSHL1f_0 = 1 #1
hSHL1s_0 = 1 #1
mSHK1_0 = 0
hSHK1_0 = 1  #1
mIRK_0 = 0 
mEGL36f_0 = 0
mEGL36m_0 = 0
mEGL36s_0 = 0
mEGL19_0 = 0
hEGL19_0 = 1 #1
mUNC2_0 = 0
hUNC2_0 = 0.25 #0.25
mCCA1_0 = 0
hCCA1_0 = 0 

t = np.arange(0, 1000, 0.5)
Istim = np.arange(-2, 14, 4)

y0 = np.array([V_0, mSHL1_0, hSHL1f_0, hSHL1s_0, mSHK1_0, hSHK1_0, mIRK_0, mEGL36f_0, mEGL36m_0, mEGL36s_0, mEGL19_0, hEGL19_0, mUNC2_0, hUNC2_0, mCCA1_0, hCCA1_0])

#TRACAGE POTENTIEL MEMBRANAIRE V
    
for I in Istim: 
    yt = odeint(f, y0, t, args=(I,))
    plt.plot(t, yt[:,0], label=I)

plt.xlabel(r"$t\;$(ms)",fontsize=20)
plt.ylabel(r"$V(t)\;$(mV)",fontsize=20)
plt.tick_params(axis='x', which='major', labelsize=20)
plt.tick_params(axis='y', which='major', labelsize=20)
#plt.ylim(-100, 10)
plt.legend()
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.show()
plt.clf()
