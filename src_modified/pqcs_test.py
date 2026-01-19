from qiskit import QuantumCircuit
import numpy as np
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import TwoLocal
from qiskit.circuit.library import CRXGate, CRYGate, CRZGate, RXGate, RYGate, RZGate

def test(num_qubits=2, num_layers=1):
    '''
    Returns a simple PQC representing a real-value matrix
    '''

    pqc = TwoLocal(num_qubits, ['ry', 'rz'], reps=num_layers, insert_barriers=True,skip_final_rotation_layer=True)
    num_params = len(pqc.parameters)
    thetas = ParameterVector('θ', num_params)
    pqc = pqc.assign_parameters(thetas)

    return pqc

def PQC1(num_qubits=2, num_layers=1):
    '''
    Returns a simple PQC representing a real-value matrix
    '''
    thetas = ParameterVector('θ', 2*num_qubits*num_layers)
    pqc = TwoLocal(num_qubits,['ry','rz'], reps=num_layers, skip_final_rotation_layer=True)

    return pqc


def PQC2(num_qubits=2,num_layers=1):
    '''
    Returns a parametrized quantum circuit
    '''
    thetas = ParameterVector('θ', 2*num_qubits*num_layers)
    pqc = QuantumCircuit(num_qubits)
    for l in range(num_layers):
        for q in range(num_qubits):
            pqc.ry(thetas[2*q+2*num_qubits*l], q)
            pqc.rz(thetas[2*q+1+2*num_qubits*l], q)
        for q in range(num_qubits-1,0,-1):
            pqc.cx(q,q-1)
        pqc.barrier()    
        
    return pqc


def PQC3(num_qubits=3,num_layers=1):
    '''
    Returns a parametrized quantum circuit
    '''
    thetas = ParameterVector('θ', (3*num_qubits-1)*num_layers)
    pqc = QuantumCircuit(num_qubits)
    for l in range(num_layers):
        for q in range(num_qubits):
            pqc.ry(thetas[2*q   + (3*num_qubits-1)*l], q)
            pqc.rz(thetas[2*q+1 + (3*num_qubits-1)*l], q)
        for q in range(num_qubits-1,0,-1):
            pqc.crz(thetas[num_qubits-1-q + (3*num_qubits-1)*l + 2*num_qubits],q,q-1)
        pqc.barrier()
    return pqc



def PQC5(num_qubits=4, num_layers=1):
    '''
    Returns a simple PQC representing a real-value matrix
    '''
  #  thetas = ParameterVector('θ', 2*num_qubits*num_layers)
  #  pqc = TwoLocal(num_qubits,['rx','rz'],fixed_angle_entangler,'linear', reps=layers, skip_final_rotation_layer=False)

    thetas = ParameterVector('θ', 2*num_qubits*num_layers)
    thetas = ParameterVector('θ', (num_qubits+1)*num_qubits*num_layers)
    pqc = QuantumCircuit(num_qubits)
    for l in range(num_layers):
        for q in range(num_qubits):
            pqc.ry(thetas[2*q   + (num_qubits+1)*num_qubits*l], q)
            pqc.rz(thetas[2*q+1 + (num_qubits+1)*num_qubits*l], q)
        '''
        for q in range(num_qubits-1,0,-1):
            pqc.crz(np.pi,num_qubits-1,q-1)
        if num_qubits == 3:
            pqc.crz(np.pi,1,0)
            pqc.crz(np.pi,1,2)
        elif num_qubits == 4:
            pqc.crz(np.pi,2,3)
            pqc.crz(np.pi,2,1)
            pqc.crz(np.pi,2,0)
            pqc.crz(np.pi,1,3)
            pqc.crz(np.pi,1,2)
            pqc.crz(np.pi,1,0)
        elif num_qubits == 5:
           pqc.crz(np.pi,3,4)
           pqc.crz(np.pi,3,2)
           pqc.crz(np.pi,3,1)
           pqc.crz(np.pi,3,0)
           pqc.crz(np.pi,2,4)
           pqc.crz(np.pi,2,3)
           pqc.crz(np.pi,2,1)
           pqc.crz(np.pi,2,0)
           pqc.crz(np.pi,1,4)
           pqc.crz(np.pi,1,3)
           pqc.crz(np.pi,1,2)
           pqc.crz(np.pi,1,0)
        for q in range(num_qubits-1,0,-1):
            pqc.crz(np.pi,0,q)
        for q in range(num_qubits):
            pqc.rx(thetas[2*q+2*num_qubits*l], q)
            pqc.rz(thetas[2*q+1+2*num_qubits*l], q)
        '''
        for qc in range(num_qubits):
            q=0
            for qt in range(num_qubits):
                if qc==qt: continue
                pqc.crz(thetas[q + (num_qubits-1)*qc + 2*num_qubits + (num_qubits+1)*num_qubits*l],num_qubits-1-qc,num_qubits-1-qt)
                pqc.barrier()
                q+=1

    return pqc


def PQC15(num_qubits=3, num_layers=2):
    '''
    Returns a simple PQC representing a real-value matrix
    '''

    thetas = ParameterVector('θ', num_qubits*num_layers)
    pqc = TwoLocal(num_qubits,['ry'], ['cx'], 'sca', reps=num_layers, skip_final_rotation_layer=True)

    print('num_layers in pqc is ',num_layers)
    return pqc

 

