# Auto-ignition one-dimensional flamelet calculation

import numpy as np
import cantera as ct
import matplotlib.pyplot as plt

# Use counterflow flame configuration
gas = ct.Solution('gri30.yaml')
gas.TP = 300, ct.one_atm

f = ct.CounterflowDiffusionFlame(gas, width=0.05)

f.fuel_inlet.mdot = 0.24
f.fuel_inlet.X = 'CH4:1'
f.fuel_inlet.T = 320

f.oxidizer_inlet.mdot = 0.72
f.oxidizer_inlet.X = 'O2:1, N2:3.76'
f.oxidizer_inlet.T = 1400

f.boundary_emissivities = (0.0, 0.0)
f.radiation_enabled = False

f.set_refine_criteria(ratio=3, slope=0.1, curve=0.1)
f.solve(loglevel=1, auto=True)

Z = f.mixture_fraction('Bilger')
#Calculate scalar dissipation rate
# chi = f.calculate_scalar_dissipation_rate()

T = f.T
OH = f.Y[f.gas.species_index('OH')]
CH2O = f.Y[f.gas.species_index('CH2O')]

# Calculate Mixture fraction
fig, axs = plt.subplots(2)

axs[0].plot(Z, T, label='T')
axs[0].set_xlabel('Z')
axs[0].set_ylabel('T')
axs[0].legend()

axs[1].plot(Z, OH, label='OH')
axs[1].plot(Z, CH2O, label='CH2O')
axs[0].set_xlabel('Z')
axs[0].set_ylabel('Y')
axs[1].legend()
plt.show()
