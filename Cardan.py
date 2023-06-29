import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import CubicSpline
import sympy as sp
import math

x = np.array([-15,-10,-5,-2,0,2, 6, 10])
y = np.array([-15,-10,-5,-2,0,2,227,419])


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
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


a=0.00189
b=0.29501
c=14.3553
d=209.8618

A=b/a
B=c/a

P=-(A*A)/3 +B

def C(I):
    return (d-cs(I))/a

def Q(I):
    return (2*A*A*A)/27 - (A*B)/3 + C(I)

def dis(I):
    return 4*P*P*P+27*Q(I)*Q(I)

def dD(I,h=1e-6):
    return (dis(I+h)-dis(I))/h


I_var=np.arange(-14,-12,0.0001)

plt.plot(I_var,dis(I_var))
plt.axhline(0, color = "k")
plt.tick_params(axis='both', which='major', labelsize=25, length=6, width=2)
plt.show()

print(P)
print(Q(0))
print(-math.sqrt((-27*Q(0)*Q(0))/(4*P*P*P)))

theta = math.asin(-math.sqrt((-27*Q(0)*Q(0))/(4*P*P*P)))
print(math.sqrt((-4*P)/3)*math.sin((theta)/3))
print(math.sqrt((-4*P)/3)*math.sin((theta)/3)-A/3)
