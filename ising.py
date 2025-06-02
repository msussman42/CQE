# https://www.youtube.com/watch?v=35ykEg3fG4c
# pip3 install qiskit?
# python3 -m venv venv_sussman
# source venv_sussman/bin/activate  (this step must be done upon reboot)
# python3 -m pip install qiskit
# pip3 install qiskit-aer
# pip3 install matplotlib
# pip3 install numpy
# pip3 install pylatexenc
# python3 hadamard_gate.py
# Import necessary libraries
#from qiskit import QuantumCircuit, Aer, execute
#from qiskit import QuantumCircuit, execute, transpile
from qiskit import QuantumCircuit
from qiskit.circuit.library import YGate,UnitaryGate
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import numpy as np

SYGate=UnitaryGate(YGate().power(1/2),label=r"$\sqrt{Y}$")
SYdgGate=UnitaryGate(SYGate.inverse(),label=r"$\sqrt{Y}^\dag$")

def generate_1d_tfim_circuit(num_qubits, num_trotter_steps, rx_angle,trotter_barriers=False,layer_barriers=False):
    qc=QuantumCircuit(num_qubits);
    for trotter_step in range(num_trotter_steps):
        add_1d_tfim_trotter_layer(qc,rx_angle,layer_barriers)
        if trotter_barriers:
            qc.barrier()
    return qc

def add_1d_tfim_trotter_layer(qc,rx_angle,layer_barriers=False):
    #adding Rzz in the even layers
    for i in range(0,qc.num_qubits-1,2):
        qc.sdg([i,i+1])
        qc.append(SYGate,[i+1])
        qc.cx(i,i+1)
        #SY dagger Gate (inverse of unitary = hermitian transpose)
        qc.append(SYdgGate,[i+1])
    if layer_barriers:
        qc.barrier()
    #adding Rzz in the odd layers
    for i in range(1,qc.num_qubits-1,2):
        qc.sdg([i,i+1])
        qc.append(SYGate,[i+1])
        qc.cx(i,i+1)
        #SY dagger Gate (inverse of unitary = hermitian transpose)
        qc.append(SYdgGate,[i+1])
    if layer_barriers:
        qc.barrier()
    qc.rx(rx_angle,list(range(qc.num_qubits)))
    if layer_barriers:
        qc.barrier()

num_qubits=6
num_trotter_steps=1
rx_angle=0.5*np.pi

qc=generate_1d_tfim_circuit(num_qubits,num_trotter_steps,rx_angle,trotter_barriers=True,layer_barriers=True)
#qc.draw(output='mpl',fold=-1)
print(qc)
input("press enter to continue...")
