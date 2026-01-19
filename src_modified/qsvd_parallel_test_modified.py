import sys; sys.path.append('src_modified')
import argparse, configparser
import numpy as np
import time,random
from numpy import sqrt
from numpy import zeros, fill_diagonal
from numpy import random, linalg
from numpy import pi, around
from qiskit.circuit.library import CRXGate, CRYGate, CRZGate, RXGate, RYGate, RZGate

from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile

from pqcs_test import PQC1
from pqcs_test import PQC2
from pqcs_test import PQC3
from pqcs_test import PQC5
from pqcs_test import PQC15

from circuit_counts import circuit_counts
from exp import expressibility

#from losses import symmetric_svd_loss
from losses2 import symmetric_svd_loss

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def mprint(string):
    if rank==0: print(string, flush=True)

def qsvd(num_qubits, num_layers, shots, maxiter, mat_seed, x_seed):
    '''
    Construct real-values symmetric M
    Reshape it into a normalized vector
    '''

    dim = pow(2, num_qubits//2)
  #  mat_rng = random.default_rng(mat_seed)
  #  m = zeros((dim,dim))
  #  for i in range(dim):
  #      for j in range(i):
  #          m[i,j] = mat_rng.uniform(low=-1.0, high=1.0)
  #          m[j,i] = m[i,j]
  #  fill_diagonal(m, 1.0)
    m = np.loadtxt("../matrix_256.txt",dtype=float)
    m_psi = m.reshape(dim*dim)
    m_psi = m_psi / linalg.norm(m_psi)

    '''
    Construct circuit encoding M
    '''
    m_qc = QuantumCircuit(num_qubits)
    m_qc.initialize(m_psi)
    m_qc = transpile(m_qc, basis_gates = ['cx', 'rz', 'ry', 'rx', 'x', 's', 'sdg', 'h','u'], optimization_level=0)

    '''
    Construct PQC representing U
    '''
    u_qc = PQC2(num_qubits//2, num_layers)
    v_qc = PQC2(num_qubits//2, num_layers)
    urs_use_PQC15 = 'N'
    
    if urs_use_PQC15 == 'N': 
        num_params = 4*(num_qubits//2) * num_layers
       # num_params = 2*(3*num_qubits//2-1)*num_layers
       # num_params = (num_qubits//2+1)*num_qubits*num_layers
    else:
        num_params = 2*(num_qubits//2) * num_layers

    '''
    Construct initial guess
    '''
#    x_seed = random.randint(1, 100)
    x_rng = random.default_rng(x_seed) # This line guarantees all ranks are generating the same initial guess x0
    x0 = 2*pi*x_rng.random(num_params)

    mprint('num_params is {}'.format(num_params))
    mprint('x0 is {}'.format(x0))
    
    if rank==0:
        uqc = transpile(u_qc, basis_gates = ['cx', 'rz', 'ry', 'rx', 'x', 's', 'sdg', 'h'], optimization_level=0)
        print('uqc is ',uqc.decompose())
        vqc = transpile(v_qc, basis_gates = ['cx', 'rz', 'ry', 'rx', 'x', 's', 'sdg', 'h'], optimization_level=0)
        print('vqc is ',vqc.decompose())

    
    '''
    We don't need this at the moment
    if rank==0: 
        counts,state,timing = circuit_counts(x0, m_qc, u_qc, v_qc, num_qubits)
        EXP = expressibility(num_qubits, num_params, m_qc, u_qc, v_qc) 
    '''

    '''
    Set up and run gradientless minimization
    args = (m_qc, u_qc, v_qc, shots, False, comm)
    objective = lambda x: symmetric_svd_loss(x, *args)
   # bounds = [(-4*pi,4*pi)]
    method='COBYLA'
    jacobian = None
    opts = {'tol':1e-7, 'rhobeg':pi, 'disp':False, 'maxiter':maxiter}
    grad_less = minimize(objective, x0, method=method, jac=jacobian, options=opts)
    mprint('Gradientless optimization\n\n{}'.format(grad_less))
 
    Set up and run gradient-based minimization
    '''
    args = (m_qc, u_qc, v_qc, shots, True, comm)
    objective = lambda x: symmetric_svd_loss(x, *args)
   # bounds = [(-4*pi,4*pi)]
    method='L-BFGS-B'
    jacobian = True
   # opts = {'ftol':1e-6, 'gtol':1e-9, 'eps':1e-2, 'disp':False, 'maxfun':len(x0)*100, 'maxiter':maxiter}
    opts = {'ftol':1e-9, 'gtol':1e-9, 'maxfun':len(x0)*100, 'maxiter':1000}
    print('len of x0 ',len(x0))
    start = time.perf_counter()
    grad_based = minimize(objective, x0, method=method, jac=jacobian, options=opts)
    end = time.perf_counter()
    mprint('\nGradient-based optimization\n\n{}'.format(grad_based))

    if rank==0:
        value2 = grad_based.x
        opt_value2 = value2.tolist()      
        counts,state,timing = circuit_counts(opt_value2, m_qc, u_qc, v_qc, num_qubits)
        print('gradient-based counts: ',counts)         

    if rank==0:
        print('opt time using gradient method: ',end-start) 
         
    if rank==0:
        counts,state,timing = circuit_counts(x0, m_qc, u_qc, v_qc, num_qubits)
        EXP = expressibility(num_qubits, num_params, m_qc, u_qc, v_qc)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-nq', '--num_qubits', type=int, default=4, help='Number of qubits')
    parser.add_argument('-nl', '--num_layers', type=int, default=1, help='Number of layers in the PQC')
    parser.add_argument('-s', '--shots', type=int, default=1000, help='Number of shots')
    parser.add_argument('-i', '--maxiter', type=int, default=100, help='Max. number of minimization iterations')
    parser.add_argument('--mat_seed', type=int, default=0, help='Matrix RNG')
    parser.add_argument('--x_seed', type=int, default=0, help='Initial guess RNG')
    parser.add_argument('--config', help='Path to configuration file')

    args = parser.parse_args()
    if args.config:
        with open(args.config, 'rb') as file:
            config_data = json.load(file)
        for key, value in config_data.items():
            if hasattr(args, key): setattr(args, key, value)
    del args.config

    qsvd(*vars(args).values())
   
