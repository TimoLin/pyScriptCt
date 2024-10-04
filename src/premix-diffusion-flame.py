# Compare the results of premixed and diffusion flames

import os
import sys

import cantera as ct
import numpy as np
import matplotlib.pyplot as plt


def premixedFlame(gas, width, phi):

    # Set the gas state
    gas.TP = 300, ct.one_atm
    gas.set_equivalence_ratio(phi, 'CH4', {'O2':1.0, 'N2':3.76})

    # Create the flame object
    flame = ct.FreeFlame(gas, width=width)
    flame.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
    flame.solve(loglevel=1, auto=True)
    
    # Extract the temperature, OH, and heat release rate
    T = flame.T
    OH = flame.X[flame.gas.species_index('OH')]
    Q = flame.heat_release_rate

    # Save the results to a CSV file
    data = np.column_stack((flame.grid, T, OH, Q))
    np.savetxt('premixFlame.csv', data, header='grid, T, OH, heat_release_rate', comments='')

    return T, OH, Q

def diffusionFlame(gas, width):
    # Solve  diffusion counter-flow flame 
    gas.TP = 300, ct.one_atm
    flame = ct.CounterflowDiffusionFlame(gas, width=width)
    flame.fuel_inlet.mdot = 0.24
    flame.fuel_inlet.X = 'CH4:1.0'
    flame.fuel_inlet.T = 300
    flame.oxidizer_inlet.mdot = 0.72
    flame.oxidizer_inlet.X = 'O2:1.0, N2:3.76'
    flame.oxidizer_inlet.T = 300
    
    flame.boundary_emissivities = (0.0, 0.0)
    flame.radiation_enabled = False
    
    flame.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
    flame.solve(loglevel=1, auto=True)

   
    # Extract the temperature, OH, and heat release rate
    T = flame.T
    OH = flame.X[flame.gas.species_index('OH')]
    Q = flame.heat_release_rate

    # Save the results to a CSV file
    data = np.column_stack((flame.grid, T, OH, Q))
    np.savetxt('diffusionFlame.csv', data, header='grid, T, OH, heat_release_rate', comments='')

    return T, OH, Q

def main():
    # Define the gas mixture
    gas = ct.Solution('gri30.yaml')
    width = 0.06
    phi = 0.6

    if "--plot" not in sys.argv:
        # Solve the premixed flame
        T1, OH1, Q1 = premixedFlame(gas, width, 0.6)
        T2, OH2, Q2 = diffusionFlame(gas, width)
    else:
        # Load the results from the CSV files
        data1 = np.genfromtxt('premixFlame.csv', delimiter=' ', names=True)
        data2 = np.genfromtxt('diffusionFlame.csv', delimiter=' ', names=True)
        T1, OH1, Q1 = data1['T'], data1['OH'], data1['heat_release_rate']
        T2, OH2, Q2 = data2['T'], data2['OH'], data2['heat_release_rate']

        # Plot the results
        fig, ax = plt.subplots(3, 1, figsize=(6/2.54, 12/2.54), sharex=True)
        for i in range(3):
            ax[i].plot(data1['grid'], data1[data1.dtype.names[i+1]], label='Premixed')
            ax[i].plot(data2['grid'], data2[data2.dtype.names[i+1]], label='Diffusion')
            ax[i].set_ylabel(data1.dtype.names[i+1])

        ax[2].set_xlabel('Position [m]')
        ax[0].legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
    
