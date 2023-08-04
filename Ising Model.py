import numpy as np
import matplotlib.pyplot as plt

print("Seán Mooney 20334066")


def generate_lattice(n): # Generate the n x n matrix to represent the lattice
    lattice = 2 * np.random.randint(2, size=(n, n)) - 1
    return lattice


def monte_carlo(state, beta, h, n):  # Metropolis MC algorithm
    for i in range(n):
        for j in range(n):
            a = np.random.randint(0, n)
            b = np.random.randint(0, n)
            s = state[a, b]  # site selected
            ut = (state[(a + 1) % n, b] + state[a, (b + 1) % n] + state[(a - 1) % n, b] + state[a, (b - 1) % n])
            de = 2 * s * (ut + h)  # change in energy

            # If statement to decide whether or not to flip spin
            if de < 0 or np.random.rand() < np.exp(-de * beta):
                state[a, b] = s*-1
    return state


def calc_energy(state, n):  # calculate the energy of the lattice
    e = 0
    for i in range(n):
        for j in range(n):
            s = state[i, j]
            ut = (state[(i + 1) % n, j] + state[i, (j + 1) % n] + state[(i - 1) % n, j] + state[i, (j - 1) % n])
            e += -ut * s
    return e


T = np.linspace(1, 8, 50)  # temperature range being investigated
N = 10  # system size
# lists to store values, numpy arrays make the graph look slightly off and I dunno why
energy_values = []  # average energies
magnetism_values = []  # average mags
e_stds = []  # standard devs of energies
for j in T:
    state = generate_lattice(N)
    for i in range(200):  # equilibrate
        state = monte_carlo(state, 1/j, 0, N)

    # values to average over
    energies = []
    magnetisms = []
    for k in range(500):
        state = monte_carlo(state, 1/j, 0, N)

        energies.append(calc_energy(state, N))
        magnetisms.append(np.sum(state))

    energy_values.append(np.average(energies))
    magnetism_values.append(np.average(magnetisms))
    e_stds.append((1/(j**2)) * np.std(np.array(energies)))

plt.scatter(T, energy_values, marker='x', color='black')
plt.xlabel("Temperature (T)")
plt.ylabel("Energy")
plt.show()

plt.scatter(T, magnetism_values, marker='x', color='black')
plt.xlabel("Temperature (T)")
plt.ylabel("Magnetization")
plt.show()

plt.scatter(T, e_stds, marker='x', color='black')
plt.xlabel("Temperature (T)")
plt.ylabel("Heat capacity")
plt.show()

# Magnetization versus magnetic field
temp = 3
magnetism_values.clear()
M = np.linspace(-5, 5, 30)
state = generate_lattice(N)
for i in M:
    for j in range(200):  # equilibrate
        state = monte_carlo(state, 1/temp, i, N)

    magnetisms = []
    for k in range(500):
        state = monte_carlo(state, 1/temp, i, N)
        magnetisms.append(np.sum(state))
    magnetism_values.append(sum(magnetisms)/len(magnetisms))

plt.scatter(M, magnetism_values, color='black')
plt.xlabel("Magnetic field")
plt.ylabel("Magnetization")
plt.show()

# heat capacity versus system size
capacity_values = []
sizes = np.arange(1, 30, 1)  # system sizes to be investigated
for i in sizes:
    state = generate_lattice(i)
    for j in range(100):  # equilibrate
        state = monte_carlo(state, 1/temp, 0, i)

    energies = []
    for k in range(150):
        state = monte_carlo(state, 1/temp, 0, i)
        energies.append(calc_energy(state, i))

    energies = np.array(energies)
    capacity_values.append(np.std(energies))

plt.show()

plt.scatter(sizes, capacity_values, marker='x', color='black')
plt.xlabel("system size")
plt.ylabel("Heat capacity")
plt.show()
