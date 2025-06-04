from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.circuit.library import QFT
from qiskit import transpile


simulator_basic = BasicSimulator()

qc = QuantumCircuit(2,2)
qc.h(1)
qc.measure(1,1)
#qc.x(0).if_test(1,1)
qc.x(0)
qc.measure(0,0)
print(qc.draw())

job = simulator_basic.run(qc,shots=1000)
result = job.result()
#print(result)
counts = result.get_counts()
print(counts)

job = simulator_basic.run(qc,shots=10,memory=True)
result = job.result()
samples = result.get_memory()
print(samples)
# These numbers show result after each shot for this circuit

#3 qubit and 3 classical register
qc = QuantumCircuit(3,3)
qc.append(QFT(3),range(3))
qc.measure(range(3),range(3))
print(qc.draw())

#basic simulator cannot run QFD, need to transpile the circuit

qc_t=transpile(qc, backend=simulator_basic)
job = simulator_basic.run(qc_t,shots=1000)
result = job.result()
counts = result.get_counts()
print(counts)

#Basic simulator is not good for large circuit, need to find coherent states (statevector/density matrix) and need to add noise
