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
g_UNC2 =0.9 #0.9
g_CCA1= 3.1 
g_CCA1_WT = 3.1 #3.1
g_NCa = 0.05 #0.05
g_L_WT = 0.4 #0.4
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

def I_IRK(g_IRK,m,V):
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

def I_L(g_L,V):
    return g_L*(V-E_L)

# Steady-state current
def I_inf(g_CCA1, g_L,V):
    return (I_SHL1(m_SHL1inf(V),h_SHL1inf(V),h_SHL1inf(V),V) + I_NCA(V) + I_L(g_L,V) +I_SHK1(m_SHK1inf(V),h_SHK1inf(V),V)+I_IRK(g_IRK,m_IRKinf(V),V)+I_EGL36(m_EGL36inf(V),m_EGL36inf(V),m_EGL36inf(V),V)+I_EGL19(m_EGL19inf(V),h_EGL19inf(V),V)+I_UNC2(m_UNC2inf(V),h_UNC2inf(V),V)+I_CCA1(g_CCA1,m_CCA1inf(V),h_CCA1inf(V),V))


# Courant d'équilibre du neurone Wild-Type (sans modification de ses paramètres)
def I_inf_WT(V):
    return (I_SHL1(m_SHL1inf(V),h_SHL1inf(V),h_SHL1inf(V),V) + I_NCA(V) + I_L(0.4,V) +I_SHK1(m_SHK1inf(V),h_SHK1inf(V),V)+I_IRK(0.2,m_IRKinf(V),V)+I_EGL36(m_EGL36inf(V),m_EGL36inf(V),m_EGL36inf(V),V)+I_EGL19(m_EGL19inf(V),h_EGL19inf(V),V)+I_UNC2(m_UNC2inf(V),h_UNC2inf(V),V)+I_CCA1(3.1,m_CCA1inf(V),h_CCA1inf(V),V))


# Pour calculer l'erreur
"""
Vm= np.arange(-75, -30, 1)
taille=len(Vm)

g_L_var= np.arange(0,0.6,0.01)
g_L_var_liste= g_L_var.tolist()
g_CCA1_var = np.arange(0.1,6,0.3)
taille_gCCA1=len(g_CCA1_var)
gCCA1_tab = np.zeros(taille_gCCA1)
gL_min_tab = np.zeros(taille_gCCA1)
l=int(0)

for g_CCA1 in g_CCA1_var:
    k= int(0)
    somme_erreur= np.zeros(len(g_L_var))
    for g_L in g_L_var:
        erreur= np.zeros(taille)
        #print(len(erreur))
        i = int(0)
        k = k+1
        for V in Vm:
            erreur[i] = np.abs(I_inf(g_CCA1,g_L,V)-I_inf_WT(V))
            i = i +1
        somme_erreur[k-1]=sum(erreur)
        #print(erreur)
        #plt.plot(t, yt_WT[:,0], label=I)
        #plt.plot(t, yt[:,0], label=I)
        #plt.show()
    minimum= min(somme_erreur)
    somme_erreur_liste=somme_erreur.tolist()
    indice= somme_erreur_liste.index(minimum)
    minimum_gL=g_L_var_liste[indice]
    #print("minimum{}={}".format(g_CCA1,minimum_gL))
    gCCA1_tab[l]=g_CCA1
    #print(gCCA1_tab)
    gL_min_tab[l]=minimum_gL
    #print(gL_min_tab)
    l=l+1


    plt.plot(g_L_var,somme_erreur)
    plt.xlabel(r'$g_L$')
    plt.ylabel(r'$Erreur$')
    plt.title(f'g_CCA1={g_CCA1}')
    plt.show()
    plt.clf()
    plt.plot(Vm, I_inf_WT(Vm),color='k')
    plt.plot(Vm,I_inf(g_CCA1,g_L,Vm))
    plt.axhline(0, color = "k")
    plt.xlabel(r'V (mV)$')
    plt.ylabel(r'$I_inf(V)\;(pA)$')
    plt.show()

g_L_var= np.arange(0,0.6,0.001)
g_L_var_liste= g_L_var.tolist()
"""

#Fonction qui détermine le phénotype du neurone en fonction de la valeur de gCCA1 et gL

V_phenotype_max = np.arange(-70,-55,0.1)
V_phenotype_min = np.arange(-55,-40,0.1)
V_phenotype_max_liste = V_phenotype_max.tolist()
V_phenotype_min_liste=V_phenotype_min.tolist()

def phenotype(g_CCA1,g_L):
    I_inf_liste_min=[]
    I_inf_liste_max=[]
    for V in V_phenotype_min:
        I_inf_liste_min.append(I_inf(g_CCA1,g_L,V))
    for V in V_phenotype_max:
        I_inf_liste_max.append(I_inf(g_CCA1,g_L,V))
    minimum = min(I_inf_liste_min)
    maximum = max(I_inf_liste_max)
    #print("min={}".format(minimum))
    #print("max={}".format(maximum))
    indice_min= I_inf_liste_min.index(minimum)
    V_min=V_phenotype_min_liste[indice_min]
    indice_max= I_inf_liste_max.index(maximum)
    V_max=V_phenotype_max_liste[indice_max]
    #print("Vmin={}".format(V_min))
    #print("Vmax={}".format(V_max))
    if (minimum <0 and maximum>0):
        return 3
    elif (minimum>0 and I_inf(g_CCA1,g_L,V_min-1) > minimum):
        return 2
    elif (minimum<0 and maximum<0):
        return 4
    else: return 1
   

#Traçage d'un graphique donnant le phénotype du neurone en fonction de la valeur de gCCA1 et gL

g_CCA1_var_SSC = np.arange(0,7,0.05)
g_L_var_SSC = np.arange(0,0.5,0.01)   

fig = plt.figure(figsize=(15, 8))
for g_CCA1 in g_CCA1_var_SSC:
    for g_L in g_L_var_SSC:
        if phenotype(g_CCA1,g_L)==1:
            plt.scatter(g_CCA1,g_L,color='#8BC34A')
        elif phenotype(g_CCA1,g_L)==2:
            plt.scatter(g_CCA1,g_L,color='#1E88E5')
        elif phenotype(g_CCA1,g_L)==3:
            plt.scatter(g_CCA1,g_L,color='salmon')
        elif phenotype(g_CCA1,g_L)==4:
            plt.scatter(g_CCA1,g_L,color='0.8')           
plt.scatter(0.2,0.4,color='#8BC34A',label=r'$Phenotype\;1$')
plt.scatter(1,0.45,color='#1E88E5',label=r'$Phenotype\;2$')
plt.scatter(2,0.4,color='salmon',label=r'$Phenotype\;3$')
plt.scatter(2,0.1,color='0.8',label=r'$Phenotype\;2*$')
plt.scatter(3.1,0.4,color='black',label=r'$WT$')            
plt.xlabel(r'$g_{CCA1}$',fontsize=32)
plt.ylabel(r'$g_L$',fontsize=32)
plt.legend()
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.tick_params(axis='both', which='major', labelsize=32, length=6, width=2)
plt.tight_layout()
plt.show()
