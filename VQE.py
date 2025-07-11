import numpy as np
from numpy import pi
from scipy.optimize import minimize_scalar, minimize
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import qiskit as qk
from qiskit import BasicAer, execute
from qiskit.quantum_info import SparsePauliOp
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
hamiltonian = SparsePauliOp.from_list([(\"II\", 0.5), (\"XX\", -0.5), (\"YY\", -0.5), (\"ZZ\", 0.5)])
hmat = hamiltonian.to_matrix()
min_val_for_hamiltonian = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian)
print(f\"Min value of the hamiltonian using Classical Method: {min_val_for_hamiltonian.eigenvalue}\")


