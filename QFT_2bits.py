# pip3 install qiskit
# python3 -m venv venv_sussman
# source venv_sussman/bin/activate  (this step must be done upon reboot)
# python3 -m pip install qiskit
# pip3 install qiskit-aer
# pip3 install matplotlib
# python3 QFT.py
# Import necessary libraries
use_emulator_only=True

import numpy as np
from numpy.linalg import eig
from numpy import pi
from scipy.optimize import minimize_scalar, minimize
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import qiskit as qk
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
#QiskitRuntimeService.save_account('token')
from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2, EstimatorV2, QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

backend_man = FakeManilaV2()

# ibm_brisbane, ibm_sherbrooke, ibm_torino
if (not use_emulator_only):
   backend_osk = QiskitRuntimeService(channel='ibm_quantum',token="set your own token here").get_backend('ibm_brisbane') #ibm_kyoto

pi=np.pi

num_qubits=2
q=qk.QuantumRegister(num_qubits,'q')
c=qk.ClassicalRegister(num_qubits,'c')
#|00>=(1 0 0 0)=|0>
#|01>=(0 1 0 0)=|1>
#|10>=(0 0 1 0)=|2>
#|11>=(0 0 0 1)=|3>
#initially our state is 1|00> + 0|01> + 0|10> + 0|11>  (i.e. the following pieces of data 1,0,0,0)
qc = qk.QuantumCircuit(q, c)
#now our state is 0|00> + 0|01> + 1|10> + 0|11>  (i.e. the following pieces of data 0,0,1,0)
qc.x(q[1])
#W=e^{i pi/2}=i
#yk=(1/2) (x0 W^{0k} + x1 W^{1k} + x2 W^{2k} + x3 W^{3k}) = (1/2) W^{2k} = (1/2)(-1)^{k}
#So, after QFT, we have: 1/2, -1/2, 1/2, -1/2 i.e.
#(1/2)( |00> - |01> + |10> - |11> )
QFT_code=QFT(num_qubits=num_qubits,do_swaps=True,inverse=False,insert_barriers=False,name='qft')
qc.append(QFT_code,range(num_qubits))
qc.measure(q,c)
print("\n QFT Output")
print("-------------")

print(qc.decompose(reps=4).draw())
backend_aer=AerSimulator()

transpiled_qc=transpile(qc,backend_aer)
result_aer=backend_aer.run(transpiled_qc,shots=9000).result()

#transpiled_qc=transpile(qc,backend_man)
#result_aer=backend_man.run(transpiled_qc,shots=1000).result()

print(result_aer.get_counts(qc))

q=qk.QuantumRegister(num_qubits,'q')
c=qk.ClassicalRegister(num_qubits,'c')
qc = qk.QuantumCircuit(q, c)
qc.x(q[1])
QFT_code=QFT(num_qubits=num_qubits,do_swaps=True,inverse=False,insert_barriers=True,name='qft')
qc.append(QFT_code,range(num_qubits))

QFT_code2=QFT(num_qubits=num_qubits,do_swaps=True,inverse=True,insert_barriers=True,name='qftINV')
qc.append(QFT_code2,range(num_qubits))

qc.measure(q,c)
print("\n QFT with inverse QFT Output")
print("-------------")

print(qc.decompose(reps=4).draw())
backend_aer=AerSimulator()

transpiled_qc=transpile(qc,backend_aer)
result_aer=backend_aer.run(transpiled_qc,shots=9000).result()

#transpiled_qc=transpile(qc,backend_man)
#result_aer=backend_man.run(transpiled_qc,shots=1000).result()


print(result_aer.get_counts(qc))


