#python3 -m venv venv_sussman
#from the bash shell:
#source venv_sussman/bin/activate  (this step must be done upon reboot)
#python3 -m pip install qiskit
#pip3 install qiskit-aer
#pip3 install matplotlib
#python3 hadamard_gate.py
#Transpilation is the process of rewriting a given input circuit to 
#match the topology of a specific quantum device, and optimize the 
#circuit instructions for execution on noisy quantum computers.
import numpy as np
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator as Aer
from qiskit.quantum_info import SparsePauliOp
#from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
from qiskit_aer.primitives import Estimator
from scipy.optimize import minimize
import random,math
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit.circuit.library import CRXGate, CRYGate, CRZGate,TwoLocal
import json
import multiprocessing
import os


#--------------------------------------
# read in 4x4 matrix to m from a file
#--------------------------------------
#m = np.loadtxt("/Users/x3e/Desktop/QC/QSVD/matrix.txt", dtype=float)
m = np.loadtxt("matrix_64_2.txt",dtype=float)
d = m.reshape(64)
print('m is',m)

#create 6x6 matrix and rewrite into a vector list
#m = np.random.rand(32, 32)
#d = m.reshape(1024)
#print('m is',m)

i=2
j=3
#print(m[i,j], d[8*i+j])

'''
qiskit's transpile function decomposes the quantum circuit into a set of basis gates
for visualization we can use the QuantumCircuit's draw() method
'''

# normalized d
initial_state = d/np.linalg.norm(d)
#print(initial_state)

#--------------------------------------------------
# Specify total number of qubits and layers of PQC
#--------------------------------------------------

num_qubits = 6
num_layers = 5

# initializa initial_state

circuit = QuantumCircuit(num_qubits)
circuit.initialize(initial_state)
transpiled_circuit = transpile(circuit, basis_gates = ['cx', 'rz', 'ry', 'rx', 'x', 's', 'sdg', 'h'], optimization_level=0)
print("transpiled initial_state quantum code")
print(transpiled_circuit.draw())
print("end transpiled initial_state quantum code")

simulator_aer = Aer()
#estimator = Estimator(mode=backend)
estimator = Estimator()


'''
The loss function must measure a quantum circuit constructed from
the encoding quantum circuit and the variational layer composed by the parametrized quantum circuits (PQCs) U and V
We can define a function that returns the PQC and append those to the encoding quantum circuit
'''

def PQC(num_qubits=2,layers=1):
    '''
    Returns a parametrized quantum circuit
    '''
   # thetas = ParameterVector('θ', 2*num_qubits)
    pqc = TwoLocal(num_qubits, ['rx', 'rz'], reps=layers, insert_barriers=True,skip_final_rotation_layer=True)
    num_params = len(pqc.parameters)
    thetas = ParameterVector('θ', num_params)
  #  random_angles = np.random.rand(num_params) * 2 * np.pi
  #  print('random_angles',random_angles)
    pqc = pqc.assign_parameters(thetas)
     #print(pqc.decompose()) 
    return pqc



def PQC_2(num_qubits=2,layers=num_layers):
    '''
    Returns a parametrized quantum circuit
    '''
    thetas = ParameterVector('θ', 2*num_qubits*num_layers)
    pqc = QuantumCircuit(num_qubits)
    
   # for l in range(num_layers):
   #     for q in range(num_qubits):
   #         pqc.ry(thetas[2*q+l], q)
   #         pqc.rz(thetas[2*q+1+l], q)

   # for q in range(num_qubits):
    for l in range(num_layers):
        for q in range(num_qubits): 
            pqc.ry(thetas[2*q+2*num_qubits*l], q)
            pqc.rz(thetas[2*q+1+2*num_qubits*l], q)
    return pqc

#pqc = PQC_2(num_qubits//2,num_layers)
#print(pqc.draw())




def PQC2(num_qubits=2):
    '''
    Returns a parametrized quantum circuit
    '''
    thetas = ParameterVector('θ', 2*num_qubits)
    pqc = QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        pqc.ry(thetas[2*q], q)
        pqc.rz(thetas[2*q+1], q)
    for q in range(num_qubits-1,0,-1):
        pqc.cx(q,q-1)

    return pqc



def PQC2_2(num_qubits=2,layers=1):
    '''
    Returns a parametrized quantum circuit
    '''
    thetas = ParameterVector('θ', 2*num_qubits*num_layers)
    pqc = QuantumCircuit(num_qubits)
    for l in range(layers):
        for q in range(num_qubits):
            pqc.ry(thetas[2*q+2*num_qubits*l], q)
            pqc.rz(thetas[2*q+1+2*num_qubits*l], q)
        for q in range(num_qubits-1,0,-1):
            pqc.cx(q,q-1)
    return pqc


pqc = PQC2_2(num_qubits//2,num_layers)
print("pqc=PQC2_2: ")
print(pqc.draw())
print("end pqc=PQC2_2: ")

def PQC3(num_qubits=2):
    '''
    Returns a parametrized quantum circuit
    '''
    thetas = ParameterVector('θ', 2*num_qubits)
    pqc = QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        pqc.ry(thetas[2*q], q)
        pqc.rz(thetas[2*q+1], q)
    for q in range(num_qubits-1,0,-1):
        pqc.crz(np.pi,q,q-1) 
    return pqc




def PQC3_2(num_qubits=3,layers=1):
    '''
    Returns a parametrized quantum circuit
    '''
    thetas = ParameterVector('θ', 2*num_qubits*num_layers)
    pqc = QuantumCircuit(num_qubits)
    for l in range(layers):
        for q in range(num_qubits):
            pqc.ry(thetas[2*q+2*num_qubits*l], q)
            pqc.rz(thetas[2*q+1+2*num_qubits*l], q)
        for q in range(num_qubits-1,0,-1):
            pqc.crz(np.pi,q,q-1)
    return pqc





def circuit_counts(thetas,u_pqc,v_pqc,num_qubits):
    '''
    Construct encoding circuit
    '''
    circuit = QuantumCircuit(num_qubits)
    circuit.initialize(initial_state)
    circuit.barrier()
   
    '''
    Append PQCs U and V
    The method assign_parameters fixes the values of the variaitonal parameters
    '''
     
   # circuit.append(u_pqc.assign_parameters(thetas[:]), list(i for i in range(num_qubits//2)))    
   # circuit.append(v_pqc.assign_parameters(thetas[:]), list(i+num_qubits//2 for i in range(num_qubits//2)))

    circuit.append(u_pqc.assign_parameters(thetas[:len(thetas)//2]), list(i for i in range(num_qubits//2)))
    circuit.append(v_pqc.assign_parameters(thetas[len(thetas)//2:]), list(i+num_qubits//2 for i in range(num_qubits//2))) 

#    circuit.append(PQC_2(2).assign_parameters(thetas[:num_qubits:]), list(i+num_qubits//2 for i in range(num_qubits//2)))
#    circuit.append(PQC2(2).assign_parameters(thetas[:num_qubits]), list(i for i in range(num_qubits//2)))
#    circuit.append(PQC2(2).assign_parameters(thetas[:num_qubits:]), list(i+num_qubits//2 for i in range(num_qubits//2)))
#    circuit.append(PQC3(4).assign_parameters(thetas[:2*num_qubits:]), list(i for i in range(num_qubits)))
#    circuit.append(PQC3(2).assign_parameters(thetas[:num_qubits]), list(i for i in range(num_qubits//2)))
#    circuit.append(PQC3(2).assign_parameters(thetas[:num_qubits:]), list(i+num_qubits//2 for i in range(num_qubits//2))) 
  
#    print(circuit)  
    '''
    Add measurement instruction
    '''
    # print out statevector before measurement
    state = Statevector(circuit)

    circuit.measure_all()
    circuit = transpile(circuit, basis_gates = ['cx', 'rz', 'ry', 'rx', 'x', 's', 'sdg', 'h'], optimization_level=0)
    result_simulator = simulator_aer.run(circuit,shots=1000).result()
 
    counts = result_simulator.get_counts()

    #print("Time Taken in aer simulator: {} sec".format(result_simulator.time_taken))    
    time = result_simulator.time_taken
    print('time',time)


    return circuit,counts,state,time



def test_loss(thetas, initial_state, u_pqc, v_pqc, num_qubits, gradient=False):
    '''
    Construct encoding circuit
    '''
    assert len(initial_state) == pow(2,num_qubits)
    assert u_pqc.num_qubits == num_qubits//2   
 
    circuit = QuantumCircuit(num_qubits)
    circuit.initialize(initial_state)
    circuit.barrier()

    '''
    Append PQCs U and V
    The method assign_parameters fixes the values of the variaitonal parameters
    '''
    circuit.append(u_pqc.assign_parameters(thetas[:len(thetas)//2]), list(i for i in range(num_qubits//2)))
    circuit.append(v_pqc.assign_parameters(thetas[len(thetas)//2:]), list(i+num_qubits//2 for i in range(num_qubits//2)))

    '''
    Add measurement instruction
    '''
    circuit.measure_all()
    circuit = transpile(circuit, basis_gates = ['cx', 'rz', 'ry', 'rx', 'x', 's', 'sdg', 'h'], optimization_level=0)

    observable = SparsePauliOp.from_sparse_list(
    list( ("ZZ", [i,num_qubits//2+i], -1) for i in range(num_qubits//2)),
    num_qubits=num_qubits
    )

    job = estimator.run(circuit, observable, shots=1000)
    # print(job.result().values)
    loss = job.result().values[0] + num_qubits//2    

    if gradient == False:
        return loss
    grad = []

    for i in range(len(thetas)):
        '''
        Define the shifted parameters
        '''
        
        thetas_p = thetas.copy()
        thetas_m = thetas.copy()
        thetas_p[i] += np.pi/2
        thetas_m[i] -= np.pi/2

        '''
        Construct the corresponding circuits
        and ompute the loss for each of them
        '''
        circuit = QuantumCircuit(num_qubits)
        circuit.initialize(initial_state)
        circuit.barrier()
        circuit.append(u_pqc.assign_parameters(thetas_p[:len(thetas)//2]), list(i for i in range(num_qubits//2)))
        circuit.append(v_pqc.assign_parameters(thetas_p[len(thetas)//2:]), list(i+num_qubits//2 for i in range(num_qubits//2)))

        loss_p = estimator.run(circuit, observable,shots=1000).result().values[0] + num_qubits//2

        circuit = QuantumCircuit(num_qubits)
        circuit.initialize(initial_state)
        circuit.barrier()
        circuit.append(u_pqc.assign_parameters(thetas_m[:len(thetas)//2]), list(i for i in range(num_qubits//2)))
        circuit.append(v_pqc.assign_parameters(thetas_m[len(thetas)//2:]), list(i+num_qubits//2 for i in range(num_qubits//2)))

        loss_m = estimator.run(circuit, observable, shots=1000).result().values[0] + num_qubits//2
        
        grad.append(0.5 * (loss_p - loss_m))

    return loss, grad
    




#test_thetas = [1,2,3,4,5,6,7,8]
#test_thetas = random.sample(range(31, 40), 8)
#print(test_thetas)

#------------------------------------------------------------------
# Generate initial rotated angles from a specified range (0,2pi)
#------------------------------------------------------------------

test_thetas = [random.uniform(0,2*(np.pi)) for _ in range(2*num_qubits*num_layers)]
print(test_thetas)



#-----------------------------------------------------------------
# Display PQC circuit
#-----------------------------------------------------------------

circuit,counts,state,time = circuit_counts(test_thetas,PQC3_2(layers=num_layers),PQC3_2(layers=num_layers),num_qubits)
qc = transpile(circuit, basis_gates = ['cx', 'rz', 'ry', 'rx', 'x', 's', 'sdg', 'h'], optimization_level=0)
print(qc.draw())


#-----------------------------------------------------------------
# Extract information from PQC circuit
#-----------------------------------------------------------------

vector_list=[]
time_list=[]
for i in range(51):
    test_thetas = [random.uniform(0,2*(np.pi)) for _ in range(2*num_qubits*num_layers)]
    circuit,counts,state,time = circuit_counts(test_thetas,PQC3_2(layers=num_layers),PQC3_2(layers=num_layers),num_qubits)
    qc = transpile(circuit, basis_gates = ['cx', 'rz', 'ry', 'rx', 'x', 's', 'sdg', 'h'], optimization_level=0)
    vector_list.append(state)    
    time_list.append(time)

print(len(vector_list))
#print('1st vector')
#print(vector_list[0])

overlap_list=[]

for i in range(51):
    for j in range(i+1,51):
#       print(f"({i}, {j})")
       overlap = np.dot(np.conjugate(vector_list[i]),vector_list[j])      
       overlap_list.append(overlap)

print('overlap list')
print(len(overlap_list))


fidelity_list=[]

for i in range(51):
    for j in range(i+1,51):
#       print(f"({i}, {j})")
       fidelity = state_fidelity(vector_list[i], vector_list[j])
       fidelity_list.append(fidelity)

print('fidelity list')
#print(fidelity_list)
print(len(fidelity_list))


#----------------------------------------------------------------------
# use bin method to sample fidelity and compute expressibility
#----------------------------------------------------------------------


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


num_qubits = 6
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



#----------------------------------------------------------------------
# Minimize loss_function computed based on reference paper (0 is ideal)
#----------------------------------------------------------------------

num_qubits = 6
loss,grad = test_loss(test_thetas, initial_state, PQC3_2(layers=num_layers), PQC3_2(layers=num_layers), num_qubits, gradient=True)
print('loss is',loss)
print('grad is ',grad)
print('len of grad',len(grad))
print('len of test_thetas',len(test_thetas))
print('type of test_thetas',type(test_thetas))

args = (initial_state, PQC3_2(layers=num_layers), PQC3_2(layers=num_layers), num_qubits, False)


result_opt = minimize(test_loss, x0=test_thetas, args=args, jac=False, method='COBYLA',
                      options={'tol':1e-7, 'maxiter':1000})

#result_opt = minimize(test_loss, x0=test_thetas, args=args, jac=False, method='L-BFGS-B',
#                      options={'ftol':1e-6, 'maxiter':1000,'gtol':1e-6,'maxfun': len(test_thetas)*100})
print(result_opt)
value = result_opt.x
opt_value = value.tolist()
print('opt_value is ',opt_value)
print('type of opt_value',type(opt_value))

print()

args = (initial_state, PQC3_2(layers=num_layers), PQC3_2(layers=num_layers), num_qubits, True)
result_opt2 = minimize(test_loss, x0=test_thetas, args=args, jac=True, method='L-BFGS-B',
                      options={'ftol':1e-6, 'gtol':1e-6, 'maxfun':len(test_thetas)*100, 'maxiter':1000})
print(result_opt2)
value2 = result_opt2.x
opt_value2 = value2.tolist()
print('opt_value2 is ',opt_value2)
print('type of opt_value2',type(opt_value2))
print('jac is ',result_opt2.jac)
print('type of jac',type(result_opt2.jac))
print('size of jac',len(result_opt2.jac))


#----------------------------------------------------------------------
# Retrive final eigenvalues
#----------------------------------------------------------------------

def retrive_value(thetas,u_pqc,v_pqc,num_qubits):
    
    circuit = QuantumCircuit(num_qubits)
    circuit.initialize(initial_state)
    circuit.barrier()

    '''
    Append PQCs U and V
    The method assign_parameters fixes the values of the variaitonal parameters
    '''

    circuit.append(u_pqc.assign_parameters(thetas[:len(thetas)//2]), list(i for i in range(num_qubits//2)))
    circuit.append(v_pqc.assign_parameters(thetas[len(thetas)//2:]), list(i+num_qubits//2 for i in range(num_qubits//2)))


    circuit.measure_all()
    circuit = transpile(circuit, basis_gates = ['cx', 'rz', 'ry', 'rx', 'x', 's', 'sdg', 'h'], optimization_level=0)
    result_simulator = simulator_aer.run(circuit,shots=1000).result()

    counts = result_simulator.get_counts()

    return counts
    

counts = retrive_value(opt_value,PQC3_2(layers=num_layers),PQC3_2(layers=num_layers),num_qubits)
print('counts from non-gradient is ',counts)
counts2 = retrive_value(opt_value2,PQC3_2(layers=num_layers),PQC3_2(layers=num_layers),num_qubits)
print('counts from gradient is ',counts2)

#----------------------------------------------------------------------
# Save important data into json file
#----------------------------------------------------------------------

PQC = "circuit 3"
num_states = "51"

data_to_save = {
    "size of matrix" : len(m),
    "PQC" : PQC,
    "num_qubits": num_qubits,
    "num_states":num_states,
    "num_layers":num_layers,
    "time in Aer simulations": time_list,
    "Ave of time in Aer simulations":np.mean(time_list),
    "EXP": EXP,
    "message":result_opt.message,
    "success":bool(result_opt.success),
    "minimization iterations": result_opt.nfev,
    "minimized value": result_opt.fun,
    "counts for non-gradient":counts,
    "gradient message":result_opt2.message,
    "gradient success":bool(result_opt2.success),
    "gradient minimization iterations": result_opt2.nfev,
    "gradient minimized value": result_opt2.fun,
    "counts for gradient":counts2
}

file_name = "SVD_6qubits_6.json"
with open(file_name, 'w') as json_file:
    json.dump(data_to_save, json_file, indent=4)
