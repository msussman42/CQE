# Note to make this code work
# . my_env/bin/active
# pip install qiskit==1.4.2
# pip install pyscf
# pip install qiskit_nature 
#qiskit HF code:
#https://nam04.safelinks.protection.outlook.com/?url=https%3A%2F%2Fqiskit-community.github.io%2Fqiskit-nature%2Ftutorials%2F01_electronic_structure.html&data=05%7C02%7Cmsussman%40fsu.edu%7C202570baea784711352f08ddebeff349%7Ca36450ebdb0642a78d1b026719f701e3%7C0%7C0%7C638926137994081074%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&sdata=tpidKVekr0kOfMUd%2BV7UP66Pjm4fxSEX5rp7YPhzflE%3D&reserved=0
#pyscf HF code:
# https://pyscf.org/user/scf.html 
#pyscf DFT code:
#  https://pyscf.org/user/dft.html#

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from pyscf import gto, dft, scf


#---------------------------------------------------------
# Qiskit Hartree Fock [HF] code (use HF pyscf to instruct
# one electron Schroginer Equation, then use VQE to 
# solve for GS energy using the quantum circuit)
#---------------------------------------------------------

# use pyscf to create the molecule
driver = PySCFDriver(
    atom="H 0 0 0; H 0 0 0.735",
    basis="sto3g",
    charge=0,
    spin=0,
    unit=DistanceUnit.ANGSTROM,
)

problem = driver.run()
print(problem)

hamiltonian = problem.hamiltonian

coefficients = hamiltonian.electronic_integrals
print(coefficients.alpha)

# two-electron integrals
second_q_op = hamiltonian.second_q_op()
print(second_q_op)

# nuclear repulsion energy
repul_E = hamiltonian.nuclear_repulsion_energy
print(repul_E)

# molecule infor
print(problem.molecule)
print(problem.reference_energy)
print(problem.num_particles)
print(problem.num_spatial_orbitals)
print(problem.basis)

# solver for GS energy
solver = GroundStateEigensolver(
    JordanWignerMapper(),
    NumPyMinimumEigensolver(),
)

result = solver.solve(problem)
print('qiskit HF code')
print(result)

# In order to solve DFT in QC, we need to find DFT-solver like HF used above;
# If they don't have it, we can write our own, but need ensure we have access
# to the VQE source code.

#-------------------------------------------------
# Hartree Fock in pyscf (HF)
#-------------------------------------------------
mol = gto.Mole()
mol.build(
    atom = "H 0 0 0; H 0 0 0.735",
    basis = 'sto3g',
    charge = 0,
    spin = 0,
)

mf = scf.RHF(mol)
mf.kernel()
dm1 = mf.make_rdm1()

# Now switch to the neutral atom in the septet state
mol.charge = 0
mol.spin = 0
mol.build(False,False)

mf = scf.RHF(mol)
#mf.kernel(dm0=dm1)
print('HF from pyscf')
print(mf.kernel(dm0=dm1))

#-------------------------------------------------
# Density Functional Theory in pyscf (DFT)
#-------------------------------------------------

mol_hf = gto.M(atom = 'H 0 0 0; H 0 0 0.735', basis = 'sto3g', symmetry = True)
mf_hf = dft.RKS(mol_hf)
mf_hf.xc = 'lda,vwn' # default
#mf_hf.xc = 'bp86' 
mf_hf = mf_hf.newton() # second-order algortihm

print('DFT from pyscf')
print(mf_hf.kernel())
