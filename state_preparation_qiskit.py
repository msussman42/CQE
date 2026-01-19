#python3 -m venv venv_sussman
#from the bash shell:
#source venv_sussman/bin/activate  (this step must be done upon reboot)
#python3 -m pip install qiskit
#pip3 install qiskit-aer
#pip3 install qiskit_ibm_runtime
#pip3 install matplotlib
#python3 hadamard_gate.py
#Transpilation is the process of rewriting a given input circuit to 
#match the topology of a specific quantum device, and optimize the 
#circuit instructions for execution on noisy quantum computers.
import numpy as np
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator as Aer
from qiskit.quantum_info import SparsePauliOp
#from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
from qiskit_aer.primitives import Estimator
from scipy.optimize import minimize
import random,math
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit.circuit.library import CRXGate, CRYGate, CRZGate,TwoLocal
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
import json
import multiprocessing
import os

backend_man = FakeManilaV2()

#num_qubits = 6
num_qubits = 5

matrix_size=32

#m = np.loadtxt("/Users/x3e/Desktop/QC/QSVD/matrix.txt", dtype=float)
#m = np.loadtxt("matrix_64_2.txt",dtype=float)
m = np.loadtxt("matrix_32_2.txt",dtype=float)
d = m.reshape(matrix_size)
print('m is',m)

#create 6x6 matrix and rewrite into a vector list
#m = np.random.rand(32, 32)
#d = m.reshape(1024)
#print('m is',m)

i=2
j=3
#print(m[i,j], d[8*i+j])

'''
qiskit's transpile function decomposes the quantum circuit into a set of basis gates
for visualization we can use the QuantumCircuit's draw() method
'''

# normalized d
initial_state = d/np.linalg.norm(d)
#print(initial_state)

#--------------------------------------------------
# Specify total number of qubits and layers of PQC
#--------------------------------------------------

num_layers = 5

# initializa initial_state

circuit = QuantumCircuit(num_qubits)
circuit.initialize(initial_state)
#transpiled_circuit = transpile(circuit, basis_gates = ['cx', 'rz', 'ry', 'rx', 'x', 's', 'sdg', 'h'], optimization_level=0)
transpiled_circuit = transpile(circuit, backend_man, basis_gates = ['cx', 'rz', 'ry', 'rx', 'x', 's', 'sdg', 'h'], optimization_level=0)
print("transpiled initial_state quantum code")
print(transpiled_circuit.draw())
print("end transpiled initial_state quantum code")

simulator_aer = Aer()
#estimator = Estimator(mode=backend)
estimator = Estimator()

# print out statevector before measurement
state = Statevector(circuit)

number_of_shots=10000

circuit.measure_all()
#circuit = transpile(circuit, basis_gates = ['cx', 'rz', 'ry', 'rx', 'x', 's', 'sdg', 'h'], optimization_level=0)
circuit = transpile(circuit, backend_man, basis_gates = ['cx', 'rz', 'ry', 'rx', 'x', 's', 'sdg', 'h'], optimization_level=0)
#result_simulator = simulator_aer.run(circuit,shots=number_of_shots).result()
result_simulator = backend_man.run(circuit,shots=number_of_shots).result()
 
counts = result_simulator.get_counts()
print(counts)
print("expected counts")
print("norm(initial_state)=",np.linalg.norm(initial_state))
array_size=len(initial_state)
print(array_size)
for i in range(array_size):
    print(f"{i:08b}")
    prob=initial_state[i]*initial_state[i]*number_of_shots
    print(prob) 

#print("Time Taken in aer simulator: {} sec".format(result_simulator.time_taken))    

print("num_qubits=")
print(num_qubits)
print("num_layers=")
print(num_layers)

