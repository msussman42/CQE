import numpy as np
from numpy import pi
from scipy.optimize import minimize_scalar, minimize
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
# !pip install qiskit
# !pip install qiskit_aer
import qiskit as qk
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes
# !pip install qiskit_algorithms
from qiskit_algorithms import NumPyMinimumEigensolver
# U=( 1 0 0 0; 0 0 -1 0;0 -1 0 0 ; 0 0 0 1 ) 
# 2 qubit circuit: psi=c1 e1 + c2 e2 + c3 e3 + c4 e4
# the ck are probabilities.
# U=(I1 x I2 + Z1 x Z2)/2 - (X1 x X2 + Y1 x Y2)/2
# X=(0  1;1 0)
# Y=(0  -i;i 0)
# Z=(1  0;0 -1)
# I1 x I2 = (1 0 0  0;0  1 0 0;0 0  1 0; 0 0 0 1)
# X1 x X2 = (0 0 0  1;0  0 1 0;0 1  0 0; 1 0 0 0)
# Y1 x Y2 = (0 0 0 -1;0  0 1 0;0 1  0 0;-1 0 0 0)
# Z1 x Z2 = (1 0 0  0;0 -1 0 0;0 0 -1 0; 0 0 0 1)
hamiltonian = SparsePauliOp.from_list([("II", 0.5), ("XX", -0.5), ("YY", -0.5), ("ZZ", 0.5)])
hmat = hamiltonian.to_matrix()
hmat
min_val_for_hamiltonian = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian)
min_val_for_hamiltonian.eigenvalue
print(f"Min value of the hamiltonian using Classical Method: {min_val_for_hamiltonian.eigenvalue}")
#ZZ basis measurement
# start with |00> 2 qubits, 1 classical bit
qc = qk.QuantumCircuit(2, 1)
qc.barrier()
# |00> mapped to |00> (controlled not)
#qc.cnot(0,1)
qc.cx(0,1)
# measure the most significant qubit and put it in the classical (0) bit.
qc.measure(1,0) 
qc.draw()
#XX basis measurement
#ZZ basis=|i> otimes |j> i,j=0,1
#XX basis=|++>,|+->,|-+>,|-->
#|+>=(|0>+|1>)/sqrt(2)
#|->=(|0>-|1>)/sqrt(2)
#|0>=(|+>+|->)/sqrt(2)
#|1>=(|+>-|->)/sqrt(2)
#H=(1 1;1 -1)/sqrt(2)
#H|0>=(1; 1)/sqrt(2)=(|0>+|1>)/sqrt(2) (e.g. |0> interpreted as |+>)
qc = qk.QuantumCircuit(2, 1)
qc.barrier()
# |00> mapped to (1/2) (|0>+|1>)^2=(|00>+|01>+|10>+|11>)/2
qc.h(0)
qc.h(1)
# (|00>+|01>+|10>+|11>)/2 mapped to
# (|00>+|01>+|11>+|10>)/2 
#qc.cnot(0,1)
qc.cx(0,1)



