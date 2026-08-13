import math

# Constants that we need for the conversions
m_e = 9.10938215e-31  # electron rest mass [kg]
q_e = 1.60217662e-19  # elementary charge [C]
h_b = 1.05457186e-34  # Planck's constant [Js]
k_e = 8.9875517873682e9  # electric constant [(kg*m^3)/(s^2*C^2)]
c_v = 299792458  # vacuum speed of light [m/s]
N_a = 6.0221408e23  # Avogadro's constant [1/mol]
k_b = 1.380648e-23  # Boltzmann's constant [J/K]

# mass units
kg = 1 / m_e
g = 1e-3 * kg
amu = g / N_a

# length units
m = k_e * m_e * q_e**2 / h_b**2
cm = 1e-2 * m
nm = 1e-9 * m
pm = 1e-12 * m
AA = 1e-10 * m

# energy units
J = h_b**2 / (m_e * q_e**4 * k_e**2)
eV = q_e * J
meV = 1e-3 * eV
kcal = 4184 * J
mol = N_a
kcal_mol = kcal / mol
cm_1 = 1e2 * 2 * math.pi * h_b * c_v * J

# time and frequency units
s = 1 / (h_b * J)
ps = 1e-12 * s
fs = 1e-15 * s
Hz = 1 / s

# charge and dipole
C = 1 / q_e
Cm = C * m
D = 1e-21 * C * m / c_v

# voltage and electric fields
V = J / C
V_m = V / m

# power and energy density
W_cm2 = 1e4 * 8 * math.pi * k_e / c_v * V_m**2

# temperature
K = k_b * J

__all__ = [
    "kg",
    "g",
    "amu",
    "m",
    "cm",
    "nm",
    "pm",
    "AA",
    "J",
    "eV",
    "meV",
    "kcal",
    "mol",
    "kcal_mol",
    "cm_1",
    "s",
    "ps",
    "fs",
    "Hz",
    "C",
    "Cm",
    "D",
    "V",
    "V_m",
    "W_cm2",
    "K",
]
