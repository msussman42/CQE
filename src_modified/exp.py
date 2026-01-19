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
from circuit_counts import circuit_counts



def expressibility(num_qubits,num_params,m_qc, u_qc, v_qc):
    vector_list=[]
    time_list=[]
    for i in range(51):
        test_thetas = [random.uniform(0,2*(np.pi)) for _ in range(num_params)]
   # test_thetas = [random.uniform(0,2*(np.pi)) for _ in range(num_qubits*num_layers)]
        counts,state,timing = circuit_counts(test_thetas, m_qc, u_qc, v_qc, num_qubits)
        vector_list.append(state)
        time_list.append(timing)

    print(len(vector_list))



    fidelity_list=[]

    for i in range(51):
        for j in range(i+1,51):
#           print(f"({i}, {j})")
            fidelity = state_fidelity(vector_list[i], vector_list[j])
            fidelity_list.append(fidelity)

    print('fidelity list')
    #print(fidelity_list)
    print(len(fidelity_list))

    bin_edges = [0,          0.01351351, 0.02702703, 0.04054054, 0.05405405, 0.06756757,
    0.08108108, 0.09459459, 0.10810811, 0.12162162, 0.13513514, 0.14864865,
    0.16216216, 0.17567568, 0.18918919, 0.2027027,  0.21621622, 0.22972973,
    0.24324324, 0.25675676, 0.27027027, 0.28378378, 0.2972973,  0.31081081,
    0.32432432, 0.33783784, 0.35135135, 0.36486486, 0.37837838, 0.39189189,
    0.40540541, 0.41891892, 0.43243243, 0.44594595, 0.45945946, 0.47297297,
    0.48648649, 0.5,        0.51351351, 0.52702703, 0.54054054, 0.55405405,
    0.56756757, 0.58108108, 0.59459459, 0.60810811, 0.62162162, 0.63513514,
    0.64864865, 0.66216216, 0.67567568, 0.68918919, 0.7027027,  0.71621622,
    0.72972973, 0.74324324, 0.75675676, 0.77027027, 0.78378378, 0.7972973,
    0.81081081, 0.82432432, 0.83783784, 0.85135135, 0.86486486, 0.87837838,
    0.89189189, 0.90540541, 0.91891892, 0.93243243, 0.94594595, 0.95945946,
    0.97297297, 0.98648649, 1.0        ]

    hist, bins = np.histogram(fidelity_list, bins=bin_edges)

# Print the result
    print('Bin Edges', bins)
    print('Counts', hist)
    print('len of counts',len(hist))

    mean_list =[]

    for i in range(74):
        num = bin_edges[i]+0.5*(bin_edges[i+1]-bin_edges[i])
        mean_list.append(num)

    print(mean_list)
    print(len(mean_list))


    EXP = 0
    num_bins = 74
    num = 0
    num_2 = 0
    num_3 = 0

    for i in range(74):
#    print('hist[i]',hist[i])
        num = num_bins*hist[i]/len(fidelity_list)
        num_2 = (2**num_qubits-1)*((1-mean_list[i])**(2*num_qubits-2))
        if num > 0:
#       print('num,num_2',num,num_2)
            num_3 = num*math.log(num/num_2)/num_bins
            EXP = num_3 + EXP

    print('EXP:',EXP)

    return EXP
