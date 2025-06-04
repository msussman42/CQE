# https://www.youtube.com/watch?v=35ykEg3fG4c
# Evidence for the utility of quantum computing before fault
# tolerance.
# What is more optimal? with or without fault tolerance?
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

#Pauli Y gate=(0 -i; i 0)
#Y^{1/2}  note Y^*=(Y^T)^*=(0 i;-i 0)^*=(0 -i;i 0)
#eigenvalues for Y are +1 and -1  
#eigenvector matrix X=2^{-1/2}(-i i;1 1) Y^.5=X D^{.5} X^*=.5(i+1)(1 -1;1 1) 
SYGate=UnitaryGate(YGate().power(1/2),label=r"$\sqrt{Y}$")
SYdgGate=UnitaryGate(SYGate.inverse(),label=r"$\sqrt{Y}^\dag$")

def generate_1d_tfim_circuit(num_qubits, num_trotter_steps, rx_angle,num_cl_bits=0,trotter_barriers=False,layer_barriers=False):

    if num_cl_bits==0:
        qc=QuantumCircuit(num_qubits)
    else:
        qc=QuantumCircuit(num_qubits,num_cl_bits)

    for trotter_step in range(num_trotter_steps):
        add_1d_tfim_trotter_layer(qc,rx_angle,layer_barriers)
        if trotter_barriers:
            qc.barrier()
    return qc

def add_1d_tfim_trotter_layer(qc,rx_angle,layer_barriers=False):
    #adding Rzz in the even layers
    #Sdg=(1 0; 0 -i)
    #Sdg |0>=|0>
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

def append_mirrored_1d_tfim_circuit(qc,num_qubits, num_trotter_steps, rx_angle,trotter_barriers=False,layer_barriers=False):
    for trotter_step in range(num_trotter_steps):
        add_mirrored_1d_tfim_trotter_layer(qc,rx_angle,layer_barriers)
        if trotter_barriers:
            qc.barrier()
    return qc

def add_mirrored_1d_tfim_trotter_layer(qc,rx_angle,layer_barriers=False):
    qc.rx(-rx_angle,list(range(qc.num_qubits)))

    #adding Rzz in the odd layers
    for i in range(1,qc.num_qubits-1,2):
        qc.append(SYGate,[i+1])
        qc.cx(i,i+1)
        qc.append(SYdgGate,[i+1])
        qc.s([i,i+1])

    if layer_barriers:
        qc.barrier()

    #adding Rzz in the even layers
    for i in range(0,qc.num_qubits-1,2):
        qc.append(SYGate,[i+1])
        qc.cx(i,i+1)
        qc.append(SYdgGate,[i+1])
        qc.s([i,i+1])

    if layer_barriers:
        qc.barrier()



#num_qubits=6
num_trotter_steps=1
rx_angle=0.5*np.pi

max_trotter_steps=10
num_qubits=10
measured_qubits=[4,5]

qc_list=[]
for trotter_step in range(max_trotter_steps):
    qc=generate_1d_tfim_circuit(num_qubits,trotter_step,rx_angle,num_cl_bits=len(measured_qubits),trotter_barriers=True,layer_barriers=True)
    append_mirrored_1d_tfim_circuit(qc,num_qubits,trotter_step,rx_angle,trotter_barriers=True,layer_barriers=True)
    qc.measure(measured_qubits,list(range(len(measured_qubits))))
    qc_list.append(qc)

#qc.draw(output='mpl',fold=-1)

print(qc_list[1])
simulator=AerSimulator()
result=simulator.run(qc_list[1],shots=10).result()
counts=result.get_counts(qc_list[1])
print(result)
print(counts)
plot_histogram(counts).savefig('Qiskit_plot.png')
plt.show()
input("press enter to continue...")
