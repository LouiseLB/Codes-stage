import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.integrate import odeint
from sympy import symbols, lambdify
from scipy.integrate import solve_ivp
import sympy as sp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

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


"""
#Trouver les racines du courant d'équilibre (pas utile ici)
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

I=0
tol = 1e-5  # tolérance
roots = []  # liste pour stocker les racines trouvées
Vm = np.arange(-100, 5, 0.001)
for i in range(len(Vm)-1):
    a, b = Vm[i], Vm[i+1]
    try:
        root = find_root(I, a, b, tol)
        roots.append(root)
    except ValueError:
        pass

print(roots)
print(I_inf(roots[0]+0.0001),I_inf(roots[1]),I_inf(roots[2]))
"""


#Trouver le minimum et le maximum du courant d'équilibre
V_phenotype_max = np.arange(-70,-55,0.01)
V_phenotype_min = np.arange(-55,-40,0.01)
V_phenotype_max_liste = V_phenotype_max.tolist()
V_phenotype_min_liste=V_phenotype_min.tolist()

I_inf_liste_min=[]
I_inf_liste_max=[]
for V in V_phenotype_min:
    I_inf_liste_min.append(I_inf(V))
for V in V_phenotype_max:
    I_inf_liste_max.append(I_inf(V))
minimum = min(I_inf_liste_min)
maximum = max(I_inf_liste_max)
print("min={}".format(minimum))
print("max={}".format(maximum))
indice_min= I_inf_liste_min.index(minimum)
V_min=V_phenotype_min_liste[indice_min]
indice_max= I_inf_liste_max.index(maximum)
V_max=V_phenotype_max_liste[indice_max]
print("Vmin={}".format(V_min))
print("Vmax={}".format(V_max))

"""
V_interp = np.arange(-70,20,1)

plt.plot(V_interp,h(V_interp),color='b')
plt.plot(V_interp,I_inf(V_interp),color='k')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Interpolation cubique')
plt.grid(True)
plt.show()
"""

V_data = np.array([-70,V_max,V_min+15,-27])
I_inf_data = np.array([0,maximum,minimum,0])
#V_data = np.array([-68,V_max,V_min,roots[2]])  
#I_inf_data = np.array([0,maximum,minimum,0])
#V_data = np.array([-96.2,-73.2,-65,V_min,-47.3,-38.6])  
#I_inf_data = np.array([-15,-2,2,minimum,-2,10])

# Effectuer l'interpolation
interp_func = interp1d(V_data, I_inf_data, kind='cubic')

# Obtenir les coefficients de l'interpolation
coefficients = np.polyfit(V_data, I_inf_data, 3)

# Créer une expression polynomiale à partir des coefficients
n = len(coefficients) - 1  # Degré de l'interpolation
V = symbols('V')
polynomial_expr = sum(coefficients[i] * V**(n-i) for i in range(n+1))

# Afficher l'expression analytique
print(polynomial_expr)

# Convertir l'expression polynomiale en une fonction utilisable avec numpy
polynomial_func = lambdify(V, polynomial_expr)

# Plage de valeurs pour l'interpolation
V_interp = np.linspace(V_data.min()-15, V_data.max()+5, 100)

# Évaluer la fonction interpolée pour chaque valeur de V_interp
y_interpolation = polynomial_func(V_interp)

def I_inf_tilde(V):
    return coefficients[0]*V*V*V+coefficients[1]*V*V+coefficients[2]*V+coefficients[3]

# Afficher les points de données et la courbe d'interpolation cubique
plt.scatter(V_data, I_inf_data, label='Données')
plt.plot(V_interp, y_interpolation, label='Interpolation cubique',color='r')
plt.plot(V_interp,I_inf_tilde(V_interp),color='r')
plt.plot(V_interp,I_inf(V_interp),color='k')
#plt.plot(V_interp,I_inf(V_interp),color='k')
plt.xlabel('Potentiel membranaire V')
plt.ylabel('Courant d equilibre interpolé')
plt.legend()
#plt.title('Interpolation cubique')
#plt.grid(True)
plt.show()



V_var = np.arange(0,15,1)
"""
#REGRESSION LINEAIRE

x = np.array([[-15],[0],[2], [6], [10]])
y = np.array([-15,0,2,227,419])

regression = LinearRegression()
regression.fit(x, y)

# Afficher les coefficients de régression
print("Coefficient de pente :", regression.coef_)
print("Terme constant :", regression.intercept_)

plt.scatter(x, y)
plt.plot(V_var,regression.coef_*V_var + regression.intercept_)
plt.show()

"""

print(I_inf_tilde(-85))

x = np.array([-15,-10,-5,-2,0,2, 6, 10])
y = np.array([-43.65,-10,-5,-2,0,2,227,419])


# Interpolation par spline cubique
cs = CubicSpline(x, y)

# Génération de nouveaux points pour l'interpolation
x_new = np.linspace(-15, 10, 100)
y_new = cs(x_new)

# Création de symboles pour les variables
x_sym = sp.symbols('x')

# Affichage des données d'origine
plt.plot(x, y, 'ro', label='Données d\'origine')

# Affichage de l'interpolation par spline cubique
plt.plot(x_new, y_new, label='Spline cubique')

# Configurer le graphique
plt.xlabel(r"$t\;$(ms)",fontsize=20)
plt.ylabel(r"$V(t)\;$(mV)",fontsize=20)
plt.tick_params(axis='x', which='major', labelsize=20)
plt.tick_params(axis='y', which='major', labelsize=20)
plt.legend()
plt.show()

"""
# Fonction exponentielle pour la régression
def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

# Régression exponentielle
params, params_covariance = curve_fit(exponential_func, x, y)

# Récupérer les paramètres a, b, c
a, b, c = params

# Prédictions sur les données d'entraînement
y_pred = exponential_func(x, a, b, c)

# Tracer les points de données et la courbe de régression
plt.scatter(x, y, label='Données')
plt.plot(x, y_pred, color='red', label='Régression exponentielle')

# Ajouter des étiquettes et une légende
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
print(params)

# Afficher le graphique
plt.show()

"""


#Protocole de stimulation (pour I>2, on utilise la régression trouvée précédemment pour retrouver l'amplitude correcte)
def Istim(I,t):
    if (t>=100 and t<=400):
        #if I>2: 
        #    return 64.10080302 * np.exp(I*0.20051913) - 43.917193932
        #else:
        return cs(I)
    elif (t>=650 and t<=850):
        return cs(-15)
    else:
        return cs(0)

g = np.vectorize(Istim)


def f(t,y, I):
    dy = np.zeros((1,))

    V = y[0]

    dy[0] = (1/C)*(g(I,t) -I_inf_tilde(V))
    return dy 


V_0= -70

t_span = (0, 1000)
t = np.arange(0, 1000, 1)
Istim = np.arange(-2, 14, 4)

method = 'RK45'  # Méthode de résolution (Runge-Kutta d'ordre 5(4))
atol = 1e-8     # Tolérance absolue
rtol = 1e-6     # Tolérance relative


for I in Istim: 
    #solution = solve_ivp(f, t_span, [V_0], t_eval=[V_0, t_span[1]])
    solution = solve_ivp(f, t_span, [V_0], t_eval=np.linspace(t_span[0], t_span[1], 10000),args=(I,),method=method, atol=atol, rtol=rtol)
    plt.plot(solution.t, solution.y[0], label=I)
plt.xlabel(r"$t\;$(ms)",fontsize=20)
plt.ylabel(r"$V(t)\;$(mV)",fontsize=20)
plt.tick_params(axis='x', which='major', labelsize=20)
plt.tick_params(axis='y', which='major', labelsize=20)
#plt.ylim(-100, 10)
plt.legend()
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.show()
plt.clf()

I=-2
yt = odeint(f, V_0, t, args=(I,))
plt.plot(t, yt[:,0], label=I)
plt.show()
plt.clf()


### (!) Pour I=-2 ou I=2, le potentiel membranaire ne redescend pas malgré le stimulus à -15 pA
###I=-2

I=2
yt = odeint(f, V_0, t, args=(I,))
plt.plot(t, yt[:,0], label=I)
plt.show()
plt.clf()

