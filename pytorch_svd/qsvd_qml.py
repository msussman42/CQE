import sys; sys.path.append('src')
import argparse, configparser

import numpy as np
import time
from numpy import sqrt
from numpy import zeros, fill_diagonal
from numpy import random, linalg
from numpy import pi, around

from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile

from pqcs import qml_two_local

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def mprint(string):
    if rank==0: print(string, flush=True)
def qml_qsvd(num_qubits, num_layers, shots, maxiter, mat_seed, x_seed):

    from numpy import log2
    import pennylane as qml
    import torch

    num_pqc_qubits = num_qubits//2

    '''
    Construct real-valued symmetric M
    Reshape it into a normalized vector
    '''
    dim = pow(2, num_qubits//2)
 #   mat_rng = random.default_rng(mat_seed)
 #   m = zeros((dim,dim))
 #   for i in range(dim):
 #       for j in range(i,dim):
 #           m[i,j] = mat_rng.uniform(low=-1.0, high=1.0) / 10
 #           m[j,i] = m[i,j]
 #   fill_diagonal(m, 1.0)
    m = np.loadtxt("/Users/x3e/Desktop/python_test/QC/matrix_16_2.txt",dtype=float)
    m_psi = m.reshape(dim*dim)
    m_psi = m_psi / linalg.norm(m_psi)

    '''
    Construct circuit encoding M
    Convert it to pennylane operation list
    '''
    m_qc = QuantumCircuit(num_qubits)
    m_qc.initialize(m_psi)
    m_qc = transpile(m_qc, basis_gates = ['cx', 'rz', 'ry', 'rx', 'x', 's', 'sdg', 'h','u'], optimization_level=0)
    m_qc_ops = qml.from_qiskit(m_qc)

    '''
    Construct PQC representing U
    '''
    num_pqc_params = num_layers * num_pqc_qubits
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    '''
    Construct initial guess
    '''
    x_rng = random.default_rng(x_seed)
    x0 = pi/2*x_rng.random(2*num_pqc_params)
    print(x0)
    x = torch.tensor(x0, dtype=torch.float64, device=device, requires_grad=True)

    '''
    Set up torch optimization
    '''

    dev_name = 'default.qubit'
    dev_name = 'lightning.qubit'
    qdevice = qml.device(dev_name, wires=num_qubits, shots=shots)
    #device = torch.device('cpu')

    @qml.qnode(qdevice, interface='torch', diff_method='parameter-shift')
    def loss_terms(params):
        '''
        Encode matrix
        '''
        m_qc_ops()
        '''
        Append PQCs
        '''

       # print(params[:num_pqc_params])
       # print(params[num_pqc_params:])

        #print('PQC1')
        qml_two_local(params[:num_pqc_params], num_pqc_qubits, num_layers, q0=0)
        #print('PQC2')
        qml_two_local(params[num_pqc_params:], num_pqc_qubits, num_layers, q0=num_pqc_params)

        # Define 4 MPI ranks,
        # rank i compute partial_loss = ml.expval(qml.PauliZ(i) @ qml.PauliZ(i+num_pqc_qubits))
        # terms = comm.allgather(partial_loss)

        
        return [ qml.expval(qml.PauliZ(qi) @ qml.PauliZ(qi+num_pqc_qubits)) for qi in range(num_pqc_qubits)]
    
   # print(qml.draw(loss_terms)(x0))
    drawer = qml.draw(loss_terms, show_all_wires=True)
    print(drawer(x0))
  
    optimizer = torch.optim.LBFGS([x], max_iter=maxiter, line_search_fn='strong_wolfe')

    # Early stopping
    best_loss = None
    best_x = None
    early_tol=1e-6
    early_patience=10
    no_improve = 0

    loss_list = []
    x_list = []

    start = time.perf_counter()
    for it in range(maxiter):
    
        def closure():
            optimizer.zero_grad()
            loss = 0.5 * (num_pqc_qubits - sum(loss_terms(x)))
            loss.backward()
            return loss
        loss = optimizer.step(closure)

        #optimizer.zero_grad()
        #loss = 0.5 * (num_pqc_qubits - sum(loss_terms(x)))
        #loss.backward()
        #optimizer.step()


        print(it, 'loss:', loss.item())

        loss_curr = loss.detach().clone()
        if (best_loss is None) or (loss_curr < best_loss - early_tol):
            best_loss = loss_curr
            best_x = x.detach().clone()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_patience:
                break

        loss_list.append(loss_curr)
        x_list.append(x.detach().clone().numpy())
    end = time.perf_counter()
   # print('opt time using gradient method: ',end-start)

    mprint('best_loss is '.format(best_loss))
    mprint('best_x is '.format(best_x))
    if rank==0: print('opt time ',end-start)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-nq', '--num_qubits', type=int, default=4, help='Number of qubits')
    parser.add_argument('-nl', '--num_layers', type=int, default=1, help='Number of layers in the PQC')
    parser.add_argument('-s', '--shots', type=int, default=1000, help='Number of shots')
    parser.add_argument('-i', '--maxiter', type=int, default=100, help='Max. number of minimization iterations')
    parser.add_argument('--mat_seed', type=int, default=0, help='Matrix RNG')
    parser.add_argument('--x_seed', type=int, default=0, help='Initial guess RNG')
    parser.add_argument('--config', help='Path to configuration file')
    # parser.add_argument('--pqc', type=str, default='PQC1', help='Path to configuration file')

    '''
    Example: --config test_run.json
    The code will try to open test_run.json
    and read all other arguments from there
    '''

    args = parser.parse_args()
    if args.config:
        with open(args.config, 'rb') as file:
            config_data = json.load(file)
        for key, value in config_data.items():
            if hasattr(args, key): setattr(args, key, value)
    del args.config

    qml_qsvd(*vars(args).values())
