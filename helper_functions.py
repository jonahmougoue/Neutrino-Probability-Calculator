from numpy import matmul, diagflat, dot, conj, real
from numpy.linalg import eig, inv
from sympy import Matrix, Symbol, symbols, zeros, ImmutableDenseMatrix, rot_axis1,rot_ccw_axis2,rot_axis3, I, diag, eye, shape
from sympy import E as Exp
from sympy.physics.quantum.dagger import Dagger
from scipy.linalg import expm


#Params:
#size: size of pmns matrix to be created
#symbol_dict: dictionary containing symbols for theta angles, phase angles, neutrino masses, energy, and potential
#Returns:
#PMNS matrix for 2x2, 3x3, or 4x4 case, depending on size
#https://en.wikipedia.org/wiki/Pontecorvo%E2%80%93Maki%E2%80%93Nakagawa%E2%80%93Sakata_matrix
def pmns_matrix(size: int,symbol_dict:dict) -> Matrix:

    U = zeros(size,size)
    if size == 2:
        U = (ImmutableDenseMatrix(rot_axis3(symbol_dict['theta_12'])).col_del(2).row_del(2))
    
    elif size == 3:
        
        complex_rot = rot_ccw_axis2(symbol_dict['theta_13'])
        complex_rot[0,2] *= Exp**(-I*symbol_dict['delta_13'])
        complex_rot[2,0] *= Exp**(I*symbol_dict['delta_13'])
        U = rot_axis1(symbol_dict['theta_23'])*complex_rot*rot_axis3(symbol_dict['theta_12'])
    else:
        
        R23 = rot_axis1(symbol_dict['theta_23']).row_insert(3,zeros(1,3)).col_insert(3,Matrix([0,0,0,1]))
        R12 = rot_axis3(symbol_dict['theta_12']).row_insert(3,zeros(1,3)).col_insert(3,Matrix([0,0,0,1]))
        R13 = rot_ccw_axis2(symbol_dict['theta_13']).row_insert(3,zeros(1,3)).col_insert(3,Matrix([0,0,0,1]))
        R13[2,0] *= Exp**(I*symbol_dict['delta_13'])
        R13[0,2] *= Exp**(-I*symbol_dict['delta_13'])

        R34 = rot_axis1(symbol_dict['theta_34']).row_insert(0,zeros(1,3)).col_insert(0,Matrix([1,0,0,0]))
        R24 = rot_axis1(symbol_dict['theta_24']).row_insert(2,zeros(1,3)).col_insert(2,Matrix([0,0,1,0]))
        R24[1,3] *= Exp**(-I*symbol_dict['delta_24'])
        R24[3,1] *= Exp**(I*symbol_dict['delta_24'])

        R14 = rot_ccw_axis2(symbol_dict['theta_14']).row_insert(2,zeros(1,3)).col_insert(2,Matrix([0,0,1,0]))
        R14[3,0] *= Exp**(-I*symbol_dict['delta_14'])
        R14[0,3] *= Exp**(I*symbol_dict['delta_14'])

        U = R34*R24*R14*R23*R13*R12
        
    return U

#Params:
#U: pmns matrix for the neutrino
#V: potential matrix for the neutrino
#symbols: dictionary containing symbols for theta angles, phase angles, neutrino masses, energy, and potential
#Returns:
#Hamiltonian for mixing matrix U and potential V
def hamiltonian(U:Matrix,V:Matrix,symbols:dict) -> Matrix:
    
    M = diag(0,*symbols['deltas'][0:shape(U)[0]-1])
    M *= 1/(2*symbols['Energy'])
    H = (U*M*Dagger(U))
    H = H-eye(shape(U)[0])*H[0,0] + V

    #Correction for units
    H *= 5
    
    return H

#Params
#lam_H: hamiltonian with lambdify() called on it/ allows for fast numeric calculations
#energy_value: energy of the moving neutrino
#potential_value: potential the neutrino is experiencing
#step_size: the distance the neutrino is traveling in the interval
#nu_I: The initial neutrino state
#Returns:
#nu_L: Neutrino state at distance > step_size + current distance
def move_nu(lam_H:callable,energy_value:float,potential_value:float,step_size:float,nu_I:list)->list:

    H = lam_H(energy_value,potential_value)
    EVs, EB = eig(H)
    Hd = diagflat(EVs)
    #exponential of the diagonalized hamiltonian    
    eHd = expm(Hd*step_size*(-1j))
    #Inverted Eigenbasis
    EBinv = inv(EB)
    #Converting to exponential of non-diagonalized hamiltonian
    eH = matmul(matmul(EB,  eHd), EBinv)
    #Wavefunction at L > 0
    nu_L = matmul(eH,nu_I)

    return nu_L

#Params:
#lam_H: hamiltonian with lambdify() called on it/ allows for fast numeric calculations
#energy_value: energy of the moving neutrino
#potential_value: potential the neutrino is experiencing
#step_size: the distance the neutrino is traveling in the interval
#nu_I: The initial neutrino state
#Returns:
#Real(P2) probability of nu_I beind detected as nu_F at distance > L
def calculate_probability(lam_H:callable,energy_value:float,potential_value:list,step_size:float,nu_I:list,nu_F:list,steps:int) -> float:
    i = 0
    nu_L = nu_I
    while i < steps:
        nu_L = move_nu(lam_H,energy_value,potential_value[i],step_size,nu_L)
        i+=1
    P = dot(nu_L,nu_F)
    Pconj = conj(P)
    P2 = (P*Pconj)
    return real(P2)


#Symbolic Variables
#Params:
#None
def make_symbols():

    symbol_dict = {}
    symbol_dict['A'] = Symbol(r"A",real = True,positive = True) #Potential
    symbol_dict['Energy'] = Symbol(r"E",real = True,positive = True) #Energy

    phase14, phase24, phase13 = symbols(r"\delta_{14}, \delta_{24}, \delta_{13}",real = True,positive = True) #Complex phase angles
    phases = [phase13,phase14,phase24]
    symbol_dict['phases'] = phases
    
    delta21, delta31, delta32, delta41 = symbols(r"\Delta_{21}, \Delta_{31}, \Delta_{32}, \Delta_{41}", real = True, positive = True) #differences in mi**2 between the 3 mass neutrino states
    deltas = [delta21,delta31,delta41]
    symbol_dict['deltas'] = deltas
    
    theta12, theta23, theta13, theta14, theta24, theta34 = symbols(r"\theta_{12}, \theta_{23}, \theta_{13}, \theta_{14}, \theta_{24}, \theta_{34}",
                                                               real = True, positive = True) #Mixing angles for 3x3 rotation matrix
    thetas = [theta12,theta23,theta13,theta14,theta24,theta34]
    symbol_dict['thetas'] = thetas

    
    for theta in thetas:
        symbol_dict[str(theta).lstrip('\\').replace('{','').replace('}','')] = theta
    for phase in phases:
        symbol_dict[str(phase).lstrip('\\').replace('{','').replace('}','')] = phase
    for delta in deltas:
        symbol_dict[str(delta).lstrip('\\').replace('{','').replace('}','')] = delta
    return symbol_dict

#Neutrino mixing angles and masses
#Current estimations can be found on nuFit: http://www.nu-fit.org/?q=node/8
def get_constants(symbols): #Hardcoded values since they're constant/can change if nufit publishes new data
    constants = {}
    constants[symbols['theta_12']] = 0.588
    constants[symbols['theta_13']] = 0.149
    constants[symbols['theta_23']] = 0.847
    constants[symbols['theta_14']] = 0.193
    constants[symbols['theta_24']] = 2.99
    constants[symbols['theta_34']] = 2.94
    constants[symbols['delta_14']] = 0.893
    constants[symbols['delta_13']] = 3.09
    constants[symbols['delta_24']] = 1.94
    constants[symbols['Delta_21']] = 8*10**(-5)
    constants[symbols['Delta_31']] = 2*10**(-3)
    constants[symbols['Delta_41']] = 1*10**(-2)

    return constants