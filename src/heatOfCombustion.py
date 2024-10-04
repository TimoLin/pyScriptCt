'''
Calculate Fuel's heat of combustion
Ref:
    https://cantera.org/examples/jupyter/thermo/heating_value.ipynb.html
'''
import cantera as ct

print(f"Using Cantera version: {ct.__version__}")

gas = ct.Solution("./grimech30.cti")

water = ct.Water()
# Set liquid water state, with vapor fraction x = 0
water.TQ = 298, 0
h_liquid = water.h

# Set gaseous water state, with vapor fraction x = 1
water.TQ = 298, 1
h_gas = water.h

def heating_value(fuel):
    """Returns the LHV and HHV for the specified fuel"""
    gas.TP = 298, ct.one_atm
    gas.set_equivalence_ratio(1.0, fuel, "O2:1.0")
    h1 = gas.enthalpy_mass
    Y_fuel = gas[fuel].Y[0]

    # complete combustion products
    X_products = {
        "CO2": gas.elemental_mole_fraction("C"),
        "H2O": 0.5 * gas.elemental_mole_fraction("H"),
        "N2": 0.5 * gas.elemental_mole_fraction("N"),
    }

    gas.TPX = None, None, X_products
    Y_H2O = gas["H2O"].Y[0]
    h2 = gas.enthalpy_mass
    LHV = -(h2 - h1) / Y_fuel / 1e6
    HHV = -(h2 - h1 + (h_liquid - h_gas) * Y_H2O) / Y_fuel / 1e6
    return LHV, HHV


fuels = ["H2", "CH4", "C2H6", "C3H8", "NH3", "CH3OH"]
print("fuel   LHV (MJ/kg)   HHV (MJ/kg)")
for fuel in fuels:
    LHV, HHV = heating_value(fuel)
    print(f"{fuel:8s} {LHV:7.3f}      {HHV:7.3f}")


