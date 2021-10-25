
def show_energy1(lattice):
    energies = []
    for i, point in enumerate(lattice):
        if i > 0 and i < n - 1:
            E = J * point * (lattice[i + 1] + lattice[i - 1])
            energies.append(E)

    _ = plt.hist(energies, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()

def get_E(lattice):

    energies = [0,0,0]
    for i, point in enumerate(lattice):
        if i == 0:
            E = J * point * (2 * lattice[i+1])
        elif i == n - 1:
            E = J * point * (2 * lattice[i-1])
        else:
            E = J * point * (lattice[i + 1] + lattice[i - 1])

        if E < 0: energies[0] += 1
        elif E > 0: energies[2] += 1
        else: energies[1] += 1

    return energies / np.sum(energies)


def evolve(lattice, target):

    energies = get_E(lattice)

    new_lattice = np.array(lattice)
    i = np.random.randint(0, n)
    new_lattice[i] *= -1
    new_energies = get_E(new_lattice)
    if np.linalg.norm(new_energies - target) < np.linalg.norm(energies - target):
        return new_lattice
    else:
        return lattice

