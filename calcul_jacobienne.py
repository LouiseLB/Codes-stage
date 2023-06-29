import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.sparse import csc_matrix
from scipy.optimize import approx_fprime
import sympy as sp

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
g_L = 0.4 #0.4

Vm= np.arange(-120, 0, 1)

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

#Steady-state current
def I_inf(V):
    return (I_SHL1(m_SHL1inf(V),h_SHL1inf(V),h_SHL1inf(V),V) + I_NCA(V) + I_L(V) +I_SHK1(m_SHK1inf(V),h_SHK1inf(V),V)+I_IRK(m_IRKinf(V),V)+I_EGL36(m_EGL36inf(V),m_EGL36inf(V),m_EGL36inf(V),V)+I_EGL19(m_EGL19inf(V),h_EGL19inf(V),V)+I_UNC2(m_UNC2inf(V),h_UNC2inf(V),V)+I_CCA1(m_CCA1inf(V),h_CCA1inf(V),V))

def f(y):
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

    dy[0] = 1/C*(I-I_SHL1(mSHL1,hSHL1f,hSHL1s,V) - I_NCA(V) - I_L(V) -I_SHK1(mSHK1,hSHK1,V)-I_IRK(mIRK,V)-I_EGL36(mEGL36f,mEGL36m,mEGL36s,V)-I_EGL19(mEGL19,hEGL19,V)-I_UNC2(mUNC2,hUNC2,V)-I_CCA1(mCCA1,hCCA1,V)) #dV/dt
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


def find_root(I, a, b, tol):
    fa = I_inf(a) - I
    fb = I_inf(b) - I
    if np.sign(fa) == np.sign(fb):
        raise ValueError("La fonction n'a pas de racine dans l'intervalle donné")
    while (b-a)/2 > tol:
        c = (a+b)/2
        fc = I_inf(c) - I
        if fc == 0:
            return c
        if np.sign(fc) == np.sign(fa):
            a, fa = c, fc
        else:
            b, fb = c, fc
    return (a+b)/2
   
#Fonction qui calcule la matrice jacobienne de la fonction f évaluée en x
def jacobian(f, x):
    n = x.size
    eps = np.sqrt(np.finfo(float).eps)
    jac = np.zeros((n, n))

    for i in range(n):
        x_eps = np.copy(x)
        x_eps[i] += eps
        jac[:, i] = (f(x_eps) - f(x)) / eps

    return jac

I_var=np.arange(-20,15,0.1)

#Trouve le nombre de racines du courant d'équilibre (donc le nombre de points d'équilibre)
for I in I_var:
    tol = 1e-2  # tolérance
    roots = []  # liste pour stocker les racines trouvées
    Vm = np.arange(-100, 5, 0.5)
    for i in range(len(Vm)-1):
        a, b = Vm[i], Vm[i+1]
        try:
            root = find_root(I, a, b, tol)
            roots.append(root)
        except ValueError:
            pass
    
    """
    #Affiche le nombre de points d'équilibre en fonction de la valeur de I
    if(len(roots)==1):
        print("Pour I={}, il y a un point d'équilibre qui est {}".format(I,roots))
    elif (len(roots)==2):
        print("Pour I={}, il y a 2 points d'équilibre qui sont {}".format(I,roots))
    else:
        print("Pour I={}, il y a 3 points d'équilibre qui sont {}".format(I,roots))
    """
        
    i_var_eq=np.arange(0,len(roots),1)
    i_var_vp=np.arange(0,16,1)

    #Calcul des valeurs propres de la matrice jacobienne
    for i in i_var_eq: 
        V=roots[i]
        y0 = np.array([V, m_SHL1inf(V), h_SHL1inf(V), h_SHL1inf(V), m_SHK1inf(V), h_SHK1inf(V), m_IRKinf(V), m_EGL36inf(V), m_EGL36inf(V), m_EGL36inf(V), m_EGL19inf(V), h_EGL19inf(V), m_UNC2inf(V), h_UNC2inf(V), m_CCA1inf(V), h_CCA1inf(V)])
        jac = jacobian(f, y0)
        valeurs_propres = np.linalg.eigvals(jac)

        #print("Valeurs propres de la jacobienne :", valeurs_propres)

        #Détermination de la stabilité des points d'équilibre en fonction du signe de la partie réelle des valeurs propres
        k=int(0)
        l=int(0)
        for i in i_var_vp:
            if (np.real(valeurs_propres[i])>0):
                k = k+1
            if (np.abs(np.real(valeurs_propres[i]))<0.00001):
                l= l+1 
        if(k==0 and l==0):
            #print("Le point d'équilibre V*={} est stable.".format(V))
            plt.scatter(I,V,color='blue')
        elif (l!=0):
            #print("Le point d'équilibre V*={} est un point col-noeud.".format(V))
            plt.scatter(I,V,color='#66CC00')
        else: 
            #print("Le point d'équilibre V*={} est instable.".format(V))
            plt.scatter(I,V,color='red')
#Traçage du nombre de point d'équilibre et de la stabilité en fonction de I            
plt.scatter(2.17,-63.09,color='#66CC00',s=45) 
plt.scatter(-13.8,-54,color='#66CC00',s=45)   
plt.xlabel(r"$I\;$(pA)",fontsize=25)
plt.ylabel(r"$V^*$",fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=25, length=6, width=2)
plt.tight_layout()
plt.show()  

#Courant d'équilibre
def g(V,I):
    return I-(I_SHL1(m_SHL1inf(V),h_SHL1inf(V),h_SHL1inf(V),V) + I_NCA(V) + I_L(V) +I_SHK1(m_SHK1inf(V),h_SHK1inf(V),V)+I_IRK(m_IRKinf(V),V)+I_EGL36(m_EGL36inf(V),m_EGL36inf(V),m_EGL36inf(V),V)+I_EGL19(m_EGL19inf(V),h_EGL19inf(V),V)+I_UNC2(m_UNC2inf(V),h_UNC2inf(V),V)+I_CCA1(m_CCA1inf(V),h_CCA1inf(V),V))


#Calcul de la dérivée du courant d'équilibre
def dI(V,h=1e-6):
    return (I_inf(V+h)-I_inf(V))/h

#Calcul de la dérivée seconde du courant d'équilibre
def ddI(V,h=1e-6):
    return (dI(V+h)-dI(V))/h


V_var=np.arange(-70,-45,0.1)

#Traçage de la dérivée du courant d'équilibre
plt.plot(V_var,dI(V_var))
plt.axhline(0, color = "k")
plt.tick_params(axis='both', which='major', labelsize=25, length=6, width=2)
plt.xlabel(r"$V\;$(mV)",fontsize=25)
plt.ylabel(r"$I'_inf(V)$",fontsize=25)
plt.tight_layout()
plt.show()
plt.clf()

#Traçage de la dérivée seconde du courant d'équilibre
plt.plot(V_var,ddI(V_var))
plt.axhline(0, color = "k")
plt.tick_params(axis='both', which='major', labelsize=25, length=6, width=2)
plt.xlabel(r"$V\;$(mV)",fontsize=25)
plt.ylabel(r"$I''_inf(V)$",fontsize=25)
plt.tight_layout()
plt.show()






