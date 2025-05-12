# Import necessary libraries
#from qiskit import QuantumCircuit, Aer, execute
#from qiskit import QuantumCircuit, execute, transpile
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

# Create a quantum circuit with 1 qubit and 1 classical bit
# default qubit value is |0>=(1 0) ?
circ = QuantumCircuit(1, 1)

# Apply a Hadamard gate to put the qubit in superposition
# H=(1/sqrt(2))(1 1 & 1 -1)
# |0> =(1 0)
# H|0>=(1 1)/sqrt(2)=(1/sqrt(2))|0>+(1/sqrt(2))|1>
circ.h(0)  # 0 is the qubit index

# Measure the qubit
circ.measure(0, 0)  # 0 is the qubit index, 0 is the classical bit index

# Run the circuit on a simulator
#backend = Aer.get_backend('qasm_simulator')
simulator=AerSimulator()
#from qiskit_aer.primitives import EstimatorV2

#job = execute(circ, backend, shots=1024)
#result = job.result()
result=simulator.run(circ,shots=1000000).result()
counts = result.get_counts(circ)
print(result)
print(counts)

# Plot the results
plot_histogram(counts).savefig('Qiskit_plot.png') # Save the histogram as an image file
plt.show()
