from qiskit import QuantumCircuit
import qiskit.quantum_info as qi
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt



qc = QuantumCircuit(2)
qc.h([0,1])
print(qc. draw())

phi = qi.Statevector(qc)
print(phi)

probs = phi.probabilities()
print(probs)

counts = phi.sample_counts(shots=1000)
print(counts)
plot_histogram(counts)
plt.show()

samples = phi.sample_memory(shots=10)
print(samples)


# Expectation of regarding operator
O = qi.SparsePauliOp(['ZZ','XX','II'],[0.2,-0.3,0.4])
print(O.to_operator())

O_exp = phi.expectation_value(O)
print(O_exp)


print(phi.draw('bloch'))

# Density of circuit
density = qi.DensityMatrix(qc)
print(density)


print(density.draw('city'))

density1 = qi.partial_trace(density,[0])
print(density1)

# Find unitary of a circuit
U = qi.Operator(qc)
print(U)

#This quantum infor cannot perform measure into classical bits

