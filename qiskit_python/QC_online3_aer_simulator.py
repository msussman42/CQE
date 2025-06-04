from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.circuit.library import QFT
from qiskit import transpile
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit_aer import noise
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)


#-----------------------------------------------------------------------
# Aer simulator
#-----------------------------------------------------------------------
simulator_basic = BasicSimulator()
simulator_aer = AerSimulator()

qc = QuantumCircuit(2,1)
qc.h(1)
qc.measure(1,0)
#qc.x(0).if_test(1,1)
#simulator_aer can save infor in quantum circuit, basic simulator cannot.
qc.save_statevector()
qc.x(0)
qc.measure(0,0)
print(qc.draw())

qc_aer = transpile(qc,backend=simulator_aer)

result = simulator_aer.run(qc_aer).result()
psi = result.get_statevector()
print(psi)

#----------------------------------------------------------------------------
# Efficient simulator (comparison between basic simulator and aer simulator)
#----------------------------------------------------------------------------
n = 5
qc = QuantumCircuit(n) 
qc.h(n-1)
for i in range(n-1,0,-1):
    qc.cx(i,i-1)  

qc.measure_all()
print(qc.draw())

#run with basic simulator
job = simulator_basic.run(qc,shots=1000)
result = job.result()
counts = result.get_counts()
print(counts)
print("Time Taken in basic simulator: {} sec".format(result.time_taken))

#run with aer simulator
job = simulator_aer.run(qc,shots=1000,method='statevector').result()
counts = job.get_counts()
print(counts)
print("Time Taken in aer simulator: {} sec".format(job.time_taken))


#run with aer simulator
job = simulator_aer.run(qc,shots=1000,method='matrix_product_state').result()
counts = job.get_counts()
print(counts)
print("Time Taken: {} sec".format(job.time_taken))

#-------------------------------------------------------------------------
#add noise using aer simulator
#-------------------------------------------------------------------------
prob = 0.1
error = depolarizing_error(prob,2) 

noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(error,['cx'])
basis_gates = noise_model.basis_gates

result_noisy = simulator_aer.run(qc,shots=10,noise_model=noise_model,basis_gates=basis_gates).result()
counts_noisy = result_noisy.get_counts()
print(counts_noisy)
job = simulator_aer.run(qc,shots=10).result()
counts = job.get_counts()
print(counts)
plot_histogram([counts_noisy,counts],legend=['Noise','No Noise'])
plt.show()

#use aer simulatir if: find counts/samples of large circuits (simulations are efficient); find statevector/density matrix of circuit with measurements; need to simulate circuits with noise
#not use aer simulator if: need to match output format of qiskit IBM runtime; Aer has its own sampler/estimator









