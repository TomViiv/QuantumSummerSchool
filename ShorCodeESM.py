# Shor Code ESM

# %%
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, IBMQ
from qiskit.quantum_info import random_statevector, Statevector
from numpy import pi
from qiskit.tools.monitor import job_monitor
#IBMQ.enable_account('9ca92f99294968c8d883deea6f397c27fd7e904601a7fee44f61a0a525f10d25dbbdf6487fc5be871e59002f8f23045845f3693f35f130a1ac6ff8e84630c607')
from qiskit import transpile
# Use AerSimulator
from qiskit_aer import AerSimulator
backend = AerSimulator()
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print("Chosen subject : IMPLEMENT QECC")

print("1) Encode state")

print("1) a) Set random vector :")
print("    random_vector = random_statevector(2)")
random_vector = random_statevector(2)
print("    random_vector : "+str(random_vector))
input()
print("    Is random_vector normalized ? "+ str(random_vector.is_valid()))
input()
p_0 = round(random_vector.probabilities()[0]*100,2)
p_1 = round(random_vector.probabilities()[1]*100,2)
print("    P(M(q=0))="+str(p_0)+"%")
print("    P(M(q=1))="+str(p_1)+"%")
print("------------------")
input()

print("1) b) Define Quantum Registers for circuit")
input()
q_input = QuantumRegister(1, 'variable')
q = QuantumRegister(8, "q_")
a = QuantumRegister(8, "a")
c = ClassicalRegister(8, 'c')
circ = QuantumCircuit(q[:4], q_input, q[4:], a, c)
print("    q_input = QuantumRegister(1, 'variable')")
print("    q = QuantumRegister(8, 'q_')")
print("    a = QuantumRegister(8, 'a')")
print("    c = ClassicalRegister(8, 'c')")
print("    circ = QuantumCircuit(q[:4], q_input, q[4:], a, c)")

def barrier():
    circ.barrier(q[:4],q_input,q[4:],a)

input()
print("    Initialized qubit 4 with the previously random vector :")
circ.initialize(random_vector,4)
print("    circ.initialize(random_vector,4)")

print("    Implementation of quantum circuit started.")
circ.cx(q_input[0], q[6])
circ.cx(q_input[0], q[1])
barrier()
print("    Controlled X gates added.")
circ.h(q[1])
circ.h(q_input[0])
circ.h(q[6])
print("    H gates added.")
circ.cx(q[1], q[2])
circ.cx(q_input[0], q[4])
circ.cx(q[6], q[7])
circ.cx(q[1], q[0])
circ.cx(q_input[0], q[3])
circ.cx(q[6], q[5])
barrier()
print("    Controlled X gates added.")

validatedInput = False
while(validatedInput==False) :
    error = input('    Choose Error to add on qubit 0. Type "X" | "Z" | "XZ" : ')
    if(error==""):
        print("    no error added")
        validatedInput = True
    if error=="X":
        circ.x(q[0])
        print("    Introduced a X error on qubit_0")
        validatedInput = True
    if error=="Z":
        circ.z(q[0])
        print("    Introduced a Z error on qubit_0")
        validatedInput = True
    if error=="XZ":
        circ.x(q[0])
        circ.z(q[0])
        print("    Introduced a XZ error on qubit_0")
        validatedInput = True
    else :
        print("    Please write a valid error.")
barrier()

for x in a:
    circ.h(x)
print("    Hadamard gates added on Ancilla qubits")

circ.cz(q[7],a[7])
circ.cz(q[6],a[7])
barrier()
circ.cz(q[4],a[5])
circ.cz(q_input[0],a[5])
barrier()
circ.cz(q[2],a[3])
circ.cz(q[1],a[3])
barrier()
circ.cz(q[6],a[6])
circ.cz(q[5],a[6])
barrier()
circ.cz(q_input[0],a[4])
circ.cz(q[3],a[4])
barrier()
circ.cz(q[1],a[2])
circ.cz(q[0],a[2])
barrier()
print("    Controlled Z gates added.")
for i in range(3,9):
    circ.cx(a[1], i)
barrier()
for i in range(0,6):
    circ.cx(a[0], i)
barrier()
print("    Controlled X gates addded.")
for i in range(8) :
    circ.h(a[i])
    circ.measure(a[i],c[i])
barrier()
print("    Hadamard gates added on ancilla qubits.")
print("    Measurements added on ancilla qubits.")

print(circ)

# Draw circuit in a file and open it :
"""circ.draw('mpl', filename="./circuit")
image_path = "./circuit.png"
image = mpimg.imread(image_path)
plt.imshow(image)
plt.show()"""

shots = 1000

# First we have to transpile the quantum circuit
# to the low-level QASM instructions used by the
# backend
qc_compiled = transpile(circ, backend)

# Execute the circuit on the qasm simulator.
# We've set the number of repeats of the circuit
# to be 1024, which is the default.
job_sim = backend.run(qc_compiled, shots=shots)

# Grab the results from the job.
result_sim = job_sim.result()
counts = result_sim.get_counts(qc_compiled)
print(counts)
ancilla_url = "./ancilla"
plot_histogram(counts, filename=ancilla_url)
image = mpimg.imread(ancilla_url+".png")
plt.imshow(image)
plt.show()

ancilla = max(counts)
print("    ancilla = ", ancilla)
"""
c_end = ClassicalRegister(9, 'end') #after error correction
#new_circ = transpile(circ, )

if(ancilla=='00000000'):
    print("there is no error")

if(ancilla=='00000100'):
    #Apply X error correction on line 0
    print("It was a X error on line 0.")
    circ.x(q[0])

if(ancilla=='00000001'):
    print("It was a Z error on line 0.")
    #Apply Z correction on line 0
    circ.z(q[0])

if(ancilla=='00000101'):
    print("It was XZ error on line 0.")
    #Apply X and Z correction on line 0
    circ.x(q[0])
    circ.z(q[0])


barrier()
circ.cx(q[1], q[0])
circ.cx(q_input[0], q[3])
circ.cx(q[6], q[5])
circ.cx(q[1], q[2])
circ.cx(q_input[0], q[4])
circ.cx(q[6], q[7])
circ.h(q[1])
circ.h(q_input[0])
circ.h(q[6])
barrier()
circ.cx(q_input[0], q[1])
circ.cx(q_input[0], q[6])
barrier()
circ.add_register(c_end)
for i in range(9) :
    circ.measure(i,c_end[i])



circ.draw('mpl')

# First we have to transpile the quantum circuit
# to the low-level QASM instructions used by the
# backend
qc_compiled = transpile(circ, backend)

shots = 1000
# Execute the circuit on the qasm simulator.
# We've set the number of repeats of the circuit
# to be 1024, which is the default.
job_sim = backend.run(qc_compiled, shots=shots)

# Grab the results from the job.
result_sim = job_sim.result()
counts = result_sim.get_counts(qc_compiled)
print(counts)

import re
for e in counts.keys():
    x = re.match("^000000000", e)
    if(x):
        corrected_p_0 = round(counts[e]*100/shots,2)
    else:
        corrected_p_1 = round(counts[e]*100/shots,2)

print("P(M(q=0))="+str(p_0)+"%")
print("P(M(q=1))="+str(p_1)+"%")

print("")
from tabulate import tabulate
data = [[0, p_0, corrected_p_0],
[1, p_1, corrected_p_1]]
print(tabulate(data, headers=["State", "Initial (%)", "After correction (%)"]))
plot_histogram(counts)"""



