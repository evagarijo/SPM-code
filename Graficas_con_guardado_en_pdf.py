# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 20:25:07 2024

@author: Equipo
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# Definir la ecuación
def new_equation(lambda_k):
    return np.sin(lambda_k) - lambda_k * np.cos(lambda_k)

# Función para encontrar raíces de la ecuación
def find_roots(equation, num_roots):
    roots = []
    for k in range(1, num_roots + 1):
        initial_guess = (k - 0.5) * np.pi
        root, = opt.fsolve(equation, initial_guess)
        roots.append(root)
    return roots

# Calcular las 200 primeras raíces
num_roots = 2000
lambdas = find_roots(new_equation, num_roots)
lambdas = np.array(lambdas)

def U_p(x):
    return 4.04596 + np.exp(-42.30027 * x + 16.56714) - 0.04880 * np.arctan(50.01833 * x - 26.48897) - 0.05447 * np.arctan(18.99678 * x - 12.32362) - np.exp(78.24095 * x - 78.68074)

def U_n(x):
    return 0.13966 + 0.68920 * np.exp(-49.20361 * x) + 0.41903 * np.exp(-254.40067 * x) - np.exp(49.97886 * x - 43.37888) - 0.028221 * np.arctan(22.52300 * x - 3.65328) - 0.01308 * np.arctan(28.34801 * x - 13.43960)

def dU_p_dT(x):
    return (-0.19952 + 0.92837 * x - 1.36455 * x**2 + 0.61154 * x**3)/(1-5.66148 * x +11.47636 * x**2 -9.82431 * x**3 +3.04876 * x**4)

def dU_n_dT(x):
    return (0.00527 + 3.29927 * x - 91.79326 * x**2 + 1004.91101 * x**3 - 5812.27813 * x**4 + 19329.75490 * x**5 - 37147.89470 * x**6 + 38379.18127 * x**7 - 16515.05308 * x**8) / (1 - 48.09287 * x + 1017.23480 * x**2 - 10481.80419 * x**3 + 59431.30001 * x**4 - 195881.64880 * x**5 + 374577.31520 * x**6 - 385821.16070 * x**7 + 165705.85970 * x**8)

def calculate_xj(x_ini_j, delta_j, Ds_j, R_j, t, r_j, num_terms=100):
    lambda_k = lambdas[1:num_terms+1]
    bar_rj = r_j / R_j
    sum_term = np.sum(np.sin(lambda_k * bar_rj) / (lambda_k**2 * np.sin(lambda_k)) * np.exp(-lambda_k**2 * Ds_j / R_j**2 * t))
    xj = x_ini_j + delta_j * (3 * Ds_j / R_j**2 * t + 1/10 * (5 * bar_rj**2 - 3) - 2 / bar_rj * sum_term)
    return xj

def calculate_xj_surf(x_ini_j, delta_j, Ds_j, R_j, t, num_terms=100):
    lambda_k = lambdas[1:num_terms+1]
    sum_term = np.sum(np.exp(-lambda_k**2 * Ds_j / R_j**2 * t) / lambda_k**2)
    xj_surf = x_ini_j + delta_j * (3 * Ds_j / R_j**2 * t + 1/5 - 2 * sum_term)
    return xj_surf

def calculate_V_cell(x_p_surf, x_n_surf, I, T, R_cell, k_p, k_n, S_p, S_n, c_e):
    F = 96487
    R = 8.3145
    m_p = I/( F * k_p * S_p * cs_p_max * np.sqrt(c_e) * np.sqrt(1 - x_p_surf)* np.sqrt(x_p_surf))
    m_n = I/( F * k_n * S_n * cs_n_max * np.sqrt(c_e) * np.sqrt(1 - x_n_surf)* np.sqrt(x_n_surf))

    V_cell = (U_p(x_p_surf) - U_n(x_n_surf) 
              + (2*R * T) / (F) * np.log((np.sqrt(m_p**2 + 4) + m_p) / 2) 
              + (2 * R * T) / F * np.log((np.sqrt(m_n**2 + 4) + m_n) / 2) 
              + I * R_cell)
    
    return V_cell

def plot_evolution_with_V_cell(x_ini_p, delta_p, Ds_p, R_p, r_p, x_ini_n, delta_n, Ds_n, R_n, r_n, t_max, I, T, R_cell, k_p, k_n, S_p, S_n, c_e, num_points=100, num_terms_values=[1, 6, 11, 21]):
    t_values = np.linspace(0, t_max, num_points)
    
    markers = ['o', 's', '^', 'D', '*']  # Diferentes estilos de marcadores
    line_style = '-'  # Estilo de línea
    line_width = 0.5  # Ancho de línea más fino

    # Subplot para x_{p, \text{surf}}
    plt.figure()
    for i, num_terms in enumerate(num_terms_values):
        xj_p_surf_values = np.array([calculate_xj_surf(x_ini_p, delta_p, Ds_p, R_p, t, num_terms) for t in t_values])
        xj_n_surf_values = np.array([calculate_xj_surf(x_ini_n, delta_n, Ds_n, R_n, t, num_terms) for t in t_values])
        plt.plot(t_values, xj_p_surf_values, label=f'$x_{{p,surf}}$, N={num_terms}', marker=markers[i], markersize=0.8, linestyle=line_style, linewidth=line_width)
        plt.plot(t_values, xj_n_surf_values, label=f'$x_{{n,surf}}$, N={num_terms}', marker=markers[i], markersize=0.8, linestyle=line_style, linewidth=line_width)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('SoC $x_{j,{surf}}$ j=n,p')
    plt.title('Evolución de $x_{j,{surf}}$ j=n,p a lo largo del tiempo')
    plt.legend()
    plt.grid(True)
    plt.savefig('x_p_surf_evolucion.pdf')  # Guardar el gráfico en PDF
    plt.close()

    # Subplot para V_{cell}
    plt.figure()
    for i, num_terms in enumerate(num_terms_values):
        xj_p_surf_values = np.array([calculate_xj_surf(x_ini_p, delta_p, Ds_p, R_p, t, num_terms) for t in t_values])
        xj_n_surf_values = np.array([calculate_xj_surf(x_ini_n, delta_n, Ds_n, R_n, t, num_terms) for t in t_values])
        V_cell_values = np.array([calculate_V_cell(x_p_surf, x_n_surf, I, T, R_cell, k_p, k_n, S_p, S_n, c_e) for x_p_surf, x_n_surf in zip(xj_p_surf_values, xj_n_surf_values)])
        plt.plot(t_values, V_cell_values, label=f'$V_{{cell}}$, N={num_terms}', marker=markers[i], markersize=0.8, linestyle=line_style, linewidth=line_width)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Voltaje $V_{cell}$ (V)')
    plt.title('Evolución de $V_{cell}$ a lo largo del tiempo')
    plt.legend()
    plt.grid(True)
    plt.savefig('V_cell_evolucion.pdf')  # Guardar el gráfico en PDF
    plt.close()
    
    # Subplot para U_p
    plt.figure()
    for i, num_terms in enumerate(num_terms_values):
        xj_p_surf_values = np.array([calculate_xj_surf(x_ini_p, delta_p, Ds_p, R_p, t, num_terms) for t in t_values])
        Up_values = np.array([U_p(x_p_surf) for x_p_surf in xj_p_surf_values])
        plt.plot(xj_p_surf_values, Up_values, label=f'$U_{{p}}$, N={num_terms}', marker=markers[i], markersize=0.8, linestyle=line_style, linewidth=line_width)
    plt.xlabel('$x_{p,{surf}}$')
    plt.ylabel('OPC $U_{p}$(V)')
    plt.title('Evolución del Potencial de circuito abierto (OPC)')
    plt.legend()
    plt.grid(True)
    plt.savefig('Up_evolucion.pdf')  # Guardar el gráfico en PDF
    plt.close()
    
    # Subplot para U_n
    plt.figure()
    for i, num_terms in enumerate(num_terms_values):
        xj_n_surf_values = np.array([calculate_xj_surf(x_ini_n, delta_n, Ds_n, R_n, t, num_terms) for t in t_values])
        Un_values = np.array([U_n(x_n_surf) for x_n_surf in xj_n_surf_values])
        plt.plot(xj_n_surf_values, Un_values, label=f'$U_{{n}}$, N={num_terms}', marker=markers[i], markersize=0.8, linestyle=line_style, linewidth=line_width)
    plt.xlabel('$x_{n,{surf}}$')
    plt.ylabel('OPC $U_{n}$ (V)')
    plt.title('Evolución del Potencial de circuito abierto (OPC)')
    plt.legend()
    plt.grid(True)
    plt.savefig('Un_evolucion.pdf')  # Guardar el gráfico en PDF
    plt.close()
    
    # Subplot para dU_p/dT
    plt.figure()
    for i, num_terms in enumerate(num_terms_values):
        xj_p_surf_values = np.array([calculate_xj_surf(x_ini_p, delta_p, Ds_p, R_p, t, num_terms) for t in t_values])
        dUp_values = np.array([dU_p_dT(x_p_surf) for x_p_surf in xj_p_surf_values])
        plt.plot(xj_p_surf_values, dUp_values, label=f'$dU_{{p}}$, N={num_terms}', marker=markers[i], markersize=0.8, linestyle=line_style, linewidth=line_width)
    plt.xlabel('$x_{p,{surf}}$')
    plt.ylabel('OPC $dU_{p}$(V)')
    plt.title('Evolución de la derivada del Potencial de circuito abierto (OPC)')
    plt.legend()
    plt.grid(True)
    plt.savefig('dUp_evolucion.pdf')  # Guardar el gráfico en PDF
    plt.close()
    
    # Subplot para dU_n/dT
    plt.figure()
    for i, num_terms in enumerate(num_terms_values):
        xj_n_surf_values = np.array([calculate_xj_surf(x_ini_n, delta_n, Ds_n, R_n, t, num_terms) for t in t_values])
        dUn_values = np.array([dU_n_dT(x_n_surf) for x_n_surf in xj_n_surf_values])
        plt.plot(xj_n_surf_values, dUn_values, label=f'$dU_{{n}}$, N={num_terms}', marker=markers[i], markersize=0.8, linestyle=line_style, linewidth=line_width)
    plt.xlabel('$x_{n,{surf}}$')
    plt.ylabel('OPC $dU_{n}$ (V)')
    plt.title('Evolución de la derivada del Potencial de circuito abierto (OPC)')
    plt.legend()
    plt.grid(True)
    plt.savefig('dUn_evolucion.pdf')  # Guardar el gráfico en PDF
    plt.close()

# Parámetros de ejemplo j=p
x_ini_p = 0.4952  # Concentración inicial
Ds_p = 1e-14  # Coeficiente de difusión
R_p = 8.5e-6  # Radio de la partícula
r_p = 5e-7  # Posición radial dentro de la partícula

t_max = 4200  # Tiempo máximo en segundos
I = -1.656

S_p = 1.1167
F = 96487
cs_p_max = 51410
delta_p = -I*R_p/(S_p*F*Ds_p*cs_p_max)

# Parámetros de ejemplo j=n
x_ini_n = 0.7522  # Concentración inicial
Ds_n = 3.9e-14  # Coeficiente de difusión
R_n = 12.5e-6  # Radio de la partícula
r_n = 5e-7  # Posición radial dentro de la partícula

S_n = 0.7824
cs_n_max = 31833
delta_n = I*R_n/(S_n*F*Ds_n*cs_n_max)

# Parámetros adicionales necesarios para calcular y graficar V_cell
T = 298.15  # Temperatura en Kelvin (25°C)
R_cell = 0.0162  # Resistencia de la celda en Ohmios
k_p = 6.66667e-11  # Constante cinética para el ánodo
k_n = 1.764e-11  # Constante cinética para el cátodo
c_e = 1000  # Concentración de electrolito en mol/m^3

# Graficar la evolución de V_cell y x_{p,\text{surf}} para los diferentes valores de num_terms
plot_evolution_with_V_cell(x_ini_p, delta_p, Ds_p, R_p, r_p, x_ini_n, delta_n, Ds_n, R_n, r_n, t_max, I, T, R_cell, k_p, k_n, S_p, S_n, c_e, num_terms_values=[1, 6, 11, 21])
