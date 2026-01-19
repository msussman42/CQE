from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import CRXGate, CRYGate, CRZGate,TwoLocal
from math import pi



def test_PQC(num_qubits=2, num_layers=1):
    '''
    Returns a simple PQC representing a real-value matrix
    '''

    thetas = ParameterVector('θ', 2*num_qubits*num_layers)
    pqc = TwoLocal(num_qubits,['ry','rz'], reps=num_layers, skip_final_rotation_layer=True)

    return pqc


def test_PQC2(num_qubits=2, num_layers=1):
    '''
    Returns a simple PQC representing a real-value matrix
    '''

    thetas = ParameterVector('θ', 2*num_qubits*num_layers)
    pqc = TwoLocal(num_qubits,['ry','rz'],'cx','linear', reps=num_layers, skip_final_rotation_layer=True)

    return pqc



