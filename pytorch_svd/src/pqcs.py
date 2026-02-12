from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import TwoLocal

import pennylane as qml

def test_PQC(num_qubits=2, num_layers=2):
    '''
    Returns a simple PQC representing a real-value matrix
    '''

    thetas = ParameterVector('θ', num_qubits*num_layers)
    pqc = TwoLocal(num_qubits, ['ry'], ['cz'], reps=num_layers, entanglement='full', skip_final_rotation_layer=True)

    return pqc

def qml_two_local(weights, num_qubits, num_layers, q0=0, rotation_gates=[qml.RY], entangling_gates=[qml.CZ], skip_final_rotation_layer=True):

    num_rotations = len(rotation_gates)
    num_entanglers = len(entangling_gates)
    if skip_final_rotation_layer == True:
        reps = num_layers
    else:
        reps = num_layers+1
    
#    print('reps is ',reps)
    for layer in range(reps):
        #print('l', layer)
        for qi in range(num_qubits):
            #print('q', qi)
            for j, gatej in enumerate(rotation_gates):

                #print(layer, qi, qi+q0, num_qubits*num_rotations*layer + num_rotations*qi + j )

                gatej(weights[num_qubits*num_rotations*layer + num_rotations*qi + j], wires=qi+q0)
 #               print('num_qubits*num_rotations*layer is ',num_qubits*num_rotations*layer)
 #               print('num_rotations*qi + j is ',num_rotations*qi + j)
 #               print('wires ',qi+q0)
 #       print('layer is ',layer) 
        if layer < num_layers:
            entangler = entangling_gates[layer%num_entanglers]
            for qi in range(num_qubits):
                if qi%num_qubits+q0 < (qi+1)%num_qubits+q0:
#                   print('q0 is ',q0)
                   entangler(wires=[qi%num_qubits+q0,(qi+1)%num_qubits+q0])
                 
