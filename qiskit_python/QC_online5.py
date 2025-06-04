from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account('token')

from math import pi
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.visualization import plot_distribution
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2, EstimatorV2, QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session
from qiskit_ibm_runtime.fake_provider import FakeManilaV2



backend_aer = AerSimulator ()
backend_osk = QiskitRuntimeService().get_backend('ibm_osaka') 

#--------------------------------------------------------------------------------
# Sampler
#--------------------------------------------------------------------------------

sampler_aer = SamplerV2(backend=backend_aer)
sampler_osk = SamplerV2(backend=backend_osk)

#create two circuits

qc1 = QuantumCircuit(1)
qc1.h(0)
qc1.measure_all()
print(qc1.draw())
 
theta = Parameter('theta')
phi = Parameter('phi')

qr = QuantumRegister(2,name='qr')
cr = ClassicalRegister(2,name='koala')

qc2 = QuantumCircuit(qr,cr)
qc2.rx(theta,1)
qc.cry(phi,1,0)
qc2.measure(qr,cr)
print(q2.draw())

angles = [[pi/4,pi/2],[pi/3,pi],[pi/2,pi]]

# transpile qc1 using method 1 on aer
qc1_aer = transpile(qc1,backend=backend_aer)
print(qc1_aer.draw())

#transpile qc2 using method 2 on aer
target = backend_aer.target
pm = generate_preset_pass_manager(target=target,optimization_level=1)
qc2_aer = pm.run(qc2)
print(qc2.qer.draw())

# transpile qc1 using method 1 on osk
qc1_osk = transpile(qc1,backend=backend_osk)
print(qc1_osk.draw(idle_wires=False))

# transpile qc2 using method 1 on osk
qc2_osk = transpile(qc2,backend=backend_osk)
print(qc2_osk.draw(idle_wires=False))


# Execute simulator

job_aer = sampler_aer.run([(qc1_aer,None,100),(qc2_aer,angles,1000)])
# This run can take hours
#job_osk = sampler_osk.run([(qc1_osk,None,100),(qc2_osk,angles,1000)])

result_aer = job_aer.result()
result_osk = job_osk.result()

counts_qc1_aer = result_aer[0].data.meas.get_counts()
counts_qc1_osk = result_osk[0].data.meas.get_counts()

#add all angles'results together
counts_qc2_aer = result_aer[0].data.koala.get_counts()
counts_qc2_osk = result_osk[0].data.koaka.get_counts()

#retrive specific angle's result: 0, 1, 2
counts_qc2_aer = result_aer[0].data.koala.get_counts(0)
counts_qc2_osk = result_osk[0].data.koaka.get_counts(0)



#--------------------------------------------------------------------------------
# Estimator
#--------------------------------------------------------------------------------

estimator_aer = EstimatorV2(backend=backend_aer)
estimator_osk = EstimatorV2(backend=backend_osk)


obsv_list = [[SparsePauliOp(["XX","IY"],[0.5,0.5])],[SparsePauliOp(["XX"],[1])],[SparsePauliOp(["IY"],[1])]]

# map observables to backend layout
layout= qc2.osk.layout

obsv_osk = []
for obsv in obsv_lst:
    obsv_temp = obsv[0].apply_layout(layout)
    print(obsv_temp) 
    obsv_osk.append([obsv_temp])

job2_aer = estimator_aer.run([(qc2_aer.obsv_lst,angles,0.01)])
# take hours to run below estimator_osk
#job2_osk = estimator_osk.run([(qc2_osk.obsv_lst,angles,0.01)])
job2_osk = OiskitRuntimeService().job('name_of_job')


exp_vals_aer = job2_aer.result()[0].data.evs
exp_vals_osk = job2_osk.result()[0].data.evs

# three ways of executing in runtime

# 1. send list of PUB, receive PUB results
# 2. session: classical and quantum computation 
# 3. batch: quantum portion can be paralleled 

#----------------------------------------------------------------------------------
# fake provider: run locally, mimic actual quantum processor (hardware), ensure qc
#  is working first, then run on the real QC hardware 
#---------------------------------------------------------------------------------

backend_man = FakeManilaV2()
sampler_man = SamplerV2(backend=backend_man)

qc2_man = transpile(qc2,backend=backend_man)
print(qc2_man.draw())

layout= qc2_man.layout

obsv_osk = []
for obsv in obsv_lst:
    obsv_temp = obsv[0].apply_layout(layout)
    print(obsv_temp)
    obsv_man.append([obsv_temp])

with Session(backend=backend_man) as session:
    sampler = SamplerV2(session=session)   
    job = sampler.run([(qc2_man,angles,1000)])
    pub_result = job.result()[0]
    print(f"1st circuit counts: \n {pub_result.data.koala.get_counts(0)} \n ")

    estimator = EstimatorV2(session=session)
    job = estimator.run([(qc2_man,obsv_man,angles,0.01)])
    pub_result = job.result()[0]
    print(f"Expectation values: \n {pub_result.data.evs} ")









