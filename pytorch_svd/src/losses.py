from numpy import arange
from numpy import pi

from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp

from auxiliary import generate_rank_indexes

def symmetric_svd_loss(thetas, m_qc, u_qc, shots=1000, gradient=False, comm=None):
    '''
    Computes loss function optimizing U matrix in SVD of symmetric matrix M
    Sigma = U^dagger x M x U = sum_ij U^dagger_im x M_mn x U_mj
    U must be real (use only RY rotations)

    Args:
    thetas   : variational parameters
    m_qc     : circuit encoding M
    u_qc     : parametrize quantum circuit representing U
    gradient : flag to compute gradient
    comm     : MPI communicator

    Returns:
    loss     : loss function
    grad     : gradient of the loss function
    '''

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
    Generate lists with the job idexes to be computed by each MPI rank
    Assign a subset of the total jobs to each rank
    '''
    if comm is not None:
        size = comm.Get_size()
        rank = comm.Get_rank()
        idxs = generate_rank_indexes(num_jobs, size, rank, comm)
    else:
        idxs = arange(num_jobs)

    '''
    List rank_values stores results from the jobs computed by each rank
    Job 0 computes the loss. All others compute partial derivatives
    '''
    rank_values = []

    for i in idxs:

        if i==0:
            qc = m_qc.copy()
            qc.append(u_qc.assign_parameters(thetas), list(q for q in range(num_qubits//2)))
            qc.append(u_qc.assign_parameters(thetas).inverse(), list(q+num_qubits//2 for q in range(num_qubits//2)))
            qc.measure_all()
            qc = transpile(qc, basis_gates = ['cx', 'rz', 'ry', 'rx', 'x', 's', 'sdg', 'h','u'], optimization_level=0)
            rank_values.append(estimator.run(qc, observable, shots=shots).result().values[0] + num_qubits//2)

        else:
            thetas_p = thetas.copy()
            thetas_m = thetas.copy()
            thetas_p[i-1] += pi/2
            thetas_m[i-1] -= pi/2
     
            qc_p = m_qc.copy()
            qc_p.append(u_qc.assign_parameters(thetas_p), list(q for q in range(num_qubits//2)))
            qc_p.append(u_qc.assign_parameters(thetas_p).inverse(), list(q+num_qubits//2 for q in range(num_qubits//2)))
            qc_p.measure_all()
            qc_p = transpile(qc_p, basis_gates = ['cx', 'rz', 'ry', 'rx', 'x', 's', 'sdg', 'h','u'], optimization_level=0)
            loss_p = estimator.run(qc_p, observable, shots=shots).result().values[0] + num_qubits//2

            qc_m = m_qc.copy()
            qc_m.append(u_qc.assign_parameters(thetas_m), list(q for q in range(num_qubits//2)))
            qc_m.append(u_qc.assign_parameters(thetas_m).inverse(), list(q+num_qubits//2 for q in range(num_qubits//2)))
            qc_m.measure_all()
            qc_m = transpile(qc_m, basis_gates = ['cx', 'rz', 'ry', 'rx', 'x', 's', 'sdg', 'h','u'], optimization_level=0)
            loss_m = estimator.run(qc_m, observable, shots=shots).result().values[0] + num_qubits//2

            rank_values.append(0.5 * (loss_p - loss_m))

    '''
    Gather results from each MPI rank in list values
    Flatten list of lists values into a simple list
    '''
    values = comm.allgather(rank_values)
    values = list(value for sublist in values for value in sublist)

    '''
    Return loss and gradient
    '''
    if gradient == False:
        return values[0]
    return values[0], values[1:]
