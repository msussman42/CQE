from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler,StatevectorEstimator
from qiskit.providers.basic_provider import BasicSimulator
from qiskit import transpile
from qiskit.circuit import Parameter
from math import pi
from qiskit.quantum_info import SparsePauliOp


qc = QuantumCircuit(2,2)
qc.h(1)
qc.cx(1,0)
qc.measure([1,0],[0,1])
print(qc.draw())

simulator = BasicSimulator()
sampler = StatevectorSampler()

job_simulator = simulator.run(qc,shots=1000)

job_sampler = sampler.run([qc],shots=1000) 

result_simulator = job_simulator.result()
result_sampler = job_sampler.result()

counts_simulator = result_simulator.get_counts()
print(counts_simulator)
counts_sampler = result_sampler[0].data.c.get_counts()
print(counts_sampler)



theta = Parameter('theta')
qc2 = QuantumCircuit(2)
qc2.rx(theta,1)
qc2.cx(1,0)
qc2.measure_all()
print(qc2.draw())

angles = [[pi/4],[pi/3],[pi/2]]

circuits = []
for angle in angles:
    qc_temp = qc2.assign_parameters({theta:angle[0]})
    circuits.append(qc_temp)

#print(circuits[0].draw())


#----------------------------------------------------------------
# Using simulator (basic simulator, can be measured)
#----------------------------------------------------------------

circuits_t = transpile(circuits,backend=simulator)
result_simulator = simulator.run(circuits_t,shots=1000).result()

counts_simulator = []

for i, _ in enumerate(angles):
    result_temp = result_simulator.get_counts(i)
    counts_simulator.append(result_temp)

print(counts_simulator)

#----------------------------------------------------------------
# Using sampler (statevector sampler, can be measured)
#----------------------------------------------------------------

result_sampler = sampler.run([(qc2,angles)],shots=1000).result()

counts_sampler = []

for i, _ in enumerate(angles):
    result_temp = result_sampler[0].data.meas.get_counts(i)
    counts_sampler.append(result_temp)

print(counts_sampler)


#statevector sampler cannot meaure mid-circuit measurements


#----------------------------------------------------------------
# Using estimator (statevector estimator, cannot be measured)
#----------------------------------------------------------------

theta = Parameter('theta')
qc2 = QuantumCircuit(2)
qc2.rx(theta,1)
qc2.cx(1,0)
print(qc2.draw())

obsv = [[SparsePauliOp(['XX','IY'],[0.5,0.5])],[SparsePauliOp(['XX'],[1])]]

estimator = StatevectorEstimator()

result_estimator = estimator.run([(qc2,obsv,angles,0.1)]).result()
# Two separate obvs and each has three results for each angle; evs stands for expectation values, since no measurement in qc
print(result_estimator[0].data.evs)










