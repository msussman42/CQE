from numpy import arange
from numpy import pi

from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp

from auxiliary import generate_rank_indexes

def symmetric_svd_loss(thetas, m_qc, u_qc, shots=1000, gradient=False, comm=None):
    '''
    Computes loss function optimizing U matrix in SVD of symmetric matrix M
    Sigma = U^dagger A U = sum_ij U^dagger_im A_mn U_mj
    U must be real (use only RY rotations)

    Args:
    thetas   : variational parameters
    m_qc     : circuit encoding M
    u_qc     : parametrize quantum circuit representing U
    gradient : flag to compute gradient

    Returns:
    loss     : loss function
    gradient : gradient of the loss function
    '''
    print('thetas is ',thetas)
    from qiskit_aer.primitives import Estimator

    estimator = Estimator()

    num_qubits = m_qc.num_qubits
    observable = SparsePauliOp.from_sparse_list(
        list(("ZZ", [i,num_qubits//2+i], -1) for i in range(num_qubits//2)),
        num_qubits=num_qubits
    )

    '''
    Define number of jobs to be split among MPI ranks
    '''    
    if gradient == False:
        num_jobs = 1
    else:
        num_jobs = 1 + len(thetas)

    '''
    Generate lists with the job idxs to be computed by each MPI rank
    Assigns a subset of the total jobs to each rank
    '''
    if comm is not None:
        size = comm.Get_size()
        rank = comm.Get_rank()
        idxs = generate_rank_indexes(num_jobs, size, rank, comm)
    else:
        idxs = arange(num_jobs)

    '''
    The list rank_values gathers the results from the jobs computed by each rank
    Job 0 computes the loss. All other jobs compute the partial derivatives
    '''
    rank_values = []

    for i in idxs:

        if i==0:
            qc = m_qc.copy()
            qc.append(u_qc.assign_parameters(thetas), list(i for i in range(num_qubits//2)))
            qc.append(u_qc.assign_parameters(thetas).inverse(), list(i+num_qubits//2 for i in range(num_qubits//2)))
            qc.measure_all()
            qc = transpile(qc, basis_gates = ['cx', 'rz', 'ry', 'rx', 'x', 's', 'sdg', 'h','u'], optimization_level=0)
            rank_values.append(estimator.run(qc, observable, shots=shots).result().values[0] + num_qubits//2)

        else:
            thetas_p = thetas.copy()
            thetas_m = thetas.copy()
            thetas_p[i-1] += pi/2
            thetas_m[i-1] -= pi/2
     
            qc = m_qc.copy()
            qc.append(u_qc.assign_parameters(thetas_p), list(i for i in range(num_qubits//2)))
            qc.append(u_qc.assign_parameters(thetas_p).inverse(), list(i+num_qubits//2 for i in range(num_qubits//2)))
            qc.measure_all()
            qc = transpile(qc, basis_gates = ['cx', 'rz', 'ry', 'rx', 'x', 's', 'sdg', 'h','u'], optimization_level=0)
            loss_p = estimator.run(qc, observable, shots=shots).result().values[0] + num_qubits//2

            qc = m_qc.copy()
            qc.append(u_qc.assign_parameters(thetas_m), list(i for i in range(num_qubits//2)))
            qc.append(u_qc.assign_parameters(thetas_m).inverse(), list(i+num_qubits//2 for i in range(num_qubits//2)))
            qc.measure_all()
            qc = transpile(qc, basis_gates = ['cx', 'rz', 'ry', 'rx', 'x', 's', 'sdg', 'h','u'], optimization_level=0)
            loss_m = estimator.run(qc, observable, shots=shots).result().values[0] + num_qubits//2

            rank_values.append(0.5 * (loss_p - loss_m))

    '''
    Gather results from each MPI rank
    Flatten the list of lists to be a unique list
    '''
    values = comm.allgather(rank_values)
    values = list(value for sublist in values for value in sublist)

    '''
    Return loss and gradient
    '''
    if gradient == False:
        return values[0]
    return values[0], values[1:]
