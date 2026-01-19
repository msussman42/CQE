import numpy as np
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator as Aer
#from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
from qiskit_aer.primitives import Estimator
from scipy.optimize import minimize
import random,math
from qiskit.quantum_info import Statevector, state_fidelity



def circuit_counts(thetas, m_qc, u_qc, v_qc, num_qubits):
    '''
    Construct encoding circuit
    '''
    num_qubits = m_qc.num_qubits

    qc = m_qc.copy()
    qc.append(u_qc.assign_parameters(thetas[:len(thetas)//2]), list(i for i in range(num_qubits//2)))
    qc.append(v_qc.assign_parameters(thetas[len(thetas)//2:]), list(i+num_qubits//2 for i in range(num_qubits//2)))
    state = Statevector(qc)
    qc.measure_all()
    qc = transpile(qc, basis_gates = ['cx', 'rz', 'ry', 'rx', 'x', 's', 'sdg', 'h','u'], optimization_level=0)
    simulator_aer = Aer()
    result_simulator = simulator_aer.run(qc,shots=8000).result()
    counts = result_simulator.get_counts()


    #print("Time Taken in aer simulator: {} sec".format(result_simulator.time_taken))
    timing = result_simulator.time_taken
    print('time',timing)


    return counts,state,timing


