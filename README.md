# Neutrino Probability Calculator
This code calculates probability of a neutrino oscillating between flavor states over a distance when considering a potential that may vary as a function of distance.

## Inputs

Neutrinos: Integer, Values 2-4 indicate the number of neutrino flavors in the analysis.
energy_range_start: Float, the startingpoint of energies
energy_range_end: Float, the endpoint of energies
ticks: Integer, Number of intervals energy_range will be divided into
distance: Integer, Distance neutrino travels before being detected
steps: Integer, Number of intervals distance will be divided into
starting_state: Integer, flavor state the neutrino is created with, (0,1,2,3) -> Electron, muon, tau, and sterile neutrino flavor states, respectively
ending_state: Integer, (0,1,2,3) -> flavor state the neutrino is detected with, (0,1,2,3) -> Electron, muon, tau, and sterile neutrino flavor states, respectively
potential_range: Boolean, True if displaying multiple probability curves with different constant energies, false if displaying single curve with varying potentials
potentials: list, if potential_range = False, list containing potential with value potential[i] located at point = distance/steps * i
if potential_range = True, list containing multiple potential constants to be analyzed.
