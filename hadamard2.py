# pip3 install qiskit?
# python3 -m venv venv_sussman
# source venv_sussman/bin/activate  (this step must be done upon reboot)
# python3 -m pip install qiskit
# pip3 install qiskit-aer
# pip3 install matplotlib
# python3 hadamard_gate.py
#Transpilation is the process of rewriting a given input circuit to 
#match the topology of a specific quantum device, and optimize the 
#circuit instructions for execution on noisy quantum computers.
from qiskit import transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

sim = AerSimulator()
# --------------------------
# Simulating using estimator
#---------------------------
from qiskit_aer.primitives import EstimatorV2

psi1 = transpile(RealAmplitudes(num_qubits=2, reps=2), sim, optimization_level=0)
psi2 = transpile(RealAmplitudes(num_qubits=2, reps=3), sim, optimization_level=0)

#Pauli matrices:
# X=(0 1 & 1 0)  X|0>=X( 1 0) =(0 1) =|1>
# Y=(0 -i & i 0) Y|0>=Y( 1 0) =(0 i) = 0+i(i)=-1 + 0 i=-|0>
# Z=(1 0 & 0 -1) Z|0>=Z( 1 0) = ??? 
# 5 qubit Hamiltonian H=Z1 X4 + 2 Y0 Y3
# op = SparsePauliOp.from_list([("XIIZI", 1), ("IYIIY", 2)])
# H1=I0 I1 + 2 Z0 I1 + 3 I0 X1
H1 = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
# H2=Z0 I1
H2 = SparsePauliOp.from_list([("IZ", 1)])
# H3=I0 Z1 + Z0 Z1
H3 = SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)])

# Fibonocci sequence x_n = x_n-1 + x_n-2
theta1 = [0, 1, 1, 2, 3, 5] # list of numbers
# Fibonocci sequence x_n = x_n-1 + x_n-2
theta2 = [0, 1, 1, 2, 3, 5, 8, 13] # list of numbers
# "counting numbers"
theta3 = [1, 2, 3, 4, 5, 6] # list of numbers

estimator = EstimatorV2()

# calculate [ [<psi1(theta1)|H1|psi1(theta1)>,
#              <psi1(theta3)|H3|psi1(theta3)>],
#             [<psi2(theta2)|H2|psi2(theta2)>] ]
job = estimator.run(
    [
        (psi1, [H1, H3], [theta1, theta3]),
        (psi2, H2, theta2)
    ],
    precision=0.01
)
result = job.result()
print(f"expectation values : psi1 = {result[0].data.evs}, psi2 = {result[1].data.evs}")

# --------------------------
# Simulating using sampler
# --------------------------
from qiskit_aer.primitives import SamplerV2
from qiskit import QuantumCircuit

# create a Bell circuit
# |0> --- H --- CNOT   =  (|00>+|11>)/sqrt(2)
# |0> --------- CNOT
bell = QuantumCircuit(2)
bell.h(0)
bell.cx(0, 1)
bell.measure_all()

# create two parameterized circuits
pqc = RealAmplitudes(num_qubits=2, reps=2)
pqc.measure_all()
pqc = transpile(pqc, sim, optimization_level=0)
pqc2 = RealAmplitudes(num_qubits=2, reps=3)
pqc2.measure_all()
pqc2 = transpile(pqc2, sim, optimization_level=0)

theta1 = [0, 1, 1, 2, 3, 5]
theta2 = [0, 1, 2, 3, 4, 5, 6, 7]

# initialization of the sampler
sampler = SamplerV2()

# collect 128 shots from the Bell circuit
job = sampler.run([bell], shots=128)
job_result = job.result()
print(f"counts for Bell circuit : {job_result[0].data.meas.get_counts()}")
 
# run a sampler job on the parameterized circuits
job2 = sampler.run([(pqc, theta1), (pqc2, theta2)])
job_result = job2.result()
print(f"counts for parameterized circuit : {job_result[0].data.meas.get_counts()}")

# --------------------------------------------------
# Simulating with noise model from actual hardware
# --------------------------------------------------
from qiskit_ibm_runtime import QiskitRuntimeService
provider = QiskitRuntimeService(channel='ibm_quantum', token="set your own token here")
backend = provider.get_backend("ibm_kyoto")

# create sampler from the actual backend
sampler = SamplerV2.from_backend(backend)

# run a sampler job on the parameterized circuits with noise model of the actual hardware
bell_t = transpile(bell, AerSimulator(basis_gates=["ecr", "id", "rz", "sx"]), optimization_level=0)
job3 = sampler.run([bell_t], shots=128)
job_result = job3.result()
print(f"counts for Bell circuit w/noise: {job_result[0].data.meas.get_counts()}")

