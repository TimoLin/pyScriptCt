'''
Calculate adiabatic temperature
'''

import matplotlib.pyplot as plt
import numpy as np
import cantera as ct

#gas = ct.Solution("NC12H26_Hybrid.cti")

#species = {S.name: S for S in ct.Species.listFromFile("NC12H26_Hybrid.cti")}
#complete_species = [species[S] for S in ("NC12H26", "O2", "N2", "CO2", "H2O")]

#gas = ct.Solution("grimech30.cti")
gas = ct.Solution("NC12H26_Hybrid.cti")

species = {S.name: S for S in ct.Species.listFromFile("NC12H26_Hybrid.cti")
complete_species = [species[S] for S in ("NC12H26", "O2", "N2", "CO2", "H2O")]

gas1 = ct.Solution(thermo="IdealGas", species=complete_species)

#phi = np.linspace(0.1, 2.0, 101)
#phi = np.array([0.44,0.45,0.46,0.47,0.65,0.75,0.55,0.569])
phi = np.array([0.44,0.45,0.46,0.47,0.65,0.75,0.55,0.569])
T_complete = np.zeros(phi.shape)

Temp = 300
#Pressure = 13.3*ct.one_atm
Pressure = 109000

Temp = 295
Pressure = 102000

for i in range(len(phi)):
    gas1.TP = Temp, Pressure
    gas1.set_equivalence_ratio(phi[i], "CH4", "O2:1, N2:3.76")
    gas1.equilibrate("HP")
    T_complete[i] = gas1.T
    print(phi[i],gas1.T)


#gas2 = ct.Solution(thermo="IdealGas", species=species.values())
#T_incomplete = np.zeros(phi.shape)
#for i in range(len(phi)):
    #gas2.TP = Temp, Pressure
    #gas2.set_equivalence_ratio(phi[i], "NC12H26", "O2:1, N2:3.76")
    #gas2.equilibrate("HP")
    #T_incomplete[i] = gas2.T

#plt.plot(phi, T_complete, label="complete combustion", lw=2)
#plt.plot(phi, T_incomplete, label="incomplete combustion", lw=2)
#plt.grid(True)
#plt.xlabel(r"Equivalence ratio, $\phi$")
#plt.ylabel("Temperature [K]")
#plt.legend()
#plt.show()
