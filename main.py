# import matplotlib.pyplot as plt
import numpy as np
import pygame

'''
This file implements the 2d ising model, with the sole focus on generating a state which obeys the boltzmann distribution
This is an initial draft algorithm which is not meant to be computationally efficient

algorithm outline:
Pick a random index on the lattice,
Flip the spin if the energy is positive
if not: Calculate the distribution of energies of the entire system and flip if this brings the system closer to the boltzmann state
The desired state is such that the relative amounts of each state is proportional to exp(-beta * Ei)
For the 2d ising model, there are only 5 energies associated with each spin: 4J, 2J, 0, -2J, -4J each with a different degeneracy
'''

algorithm = '2D Energy Bias'

## PYGAME IMPLEMENTATION
'''draw lattice points with -1 spin grey, +1 as cyan'''
bg_color = (25, 25, 25) # (250, 250, 250)
ScreenWidth, ScreenHeight = ScreenDims = (1100, 600)  # (1800, 920)
win = pygame.display.set_mode(ScreenDims)
pygame.display.set_caption('1D-Ising model | ' + algorithm + ' Implementation')
pygame.display.set_icon(pygame.image.load('imgs/molecules.png'))

def Poll_Events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            global running
            running = False


def redraw(lattice):
    win.fill(bg_color)
    # draw spins
    colours = [0, (0, 50, 50), (163, 0, 30)]
    r=10
    for i in range(lattice.shape[0]):
        for j in range(lattice.shape[1]):
            x = r + 2*r*i
            y = r + 2*r*j
            pygame.draw.circle(win, colours[lattice[i,j]], (x,y), r)
    pygame.display.update()

### PYGAME IMPLEMENTATION

n = 20
epsilon = 1e-7
J = 1
T = 0.01
b = 1/T

class System():
    def __init__(self):
        # Initialize lattice points, [1,1,-1,1,-1,...] randomly
        self.L = np.random.randint(0, 2, size=(n,n)) * 2 - 1

        t = [np.exp(b * 4 * J), 6*np.exp(b * 2 * J), 10 * np.exp(-b * 0), 6 * np.exp(-b * 2 * J), np.exp(-b * 4 * J)]
        self.pi = t / np.sum(t)
        print('pi:', self.pi)
        self.pidist = 1/epsilon
        self.get_E()
        print('degenarcies', self.E/self.E[0])

    @staticmethod
    def get_Eij( i, j, latt):
        s = latt[i, j]
        try:
            E = -J * s * (latt[i + 1, j] + latt[i - 1, j] + latt[i, j + 1] + latt[i, j - 1])
        except:
            E = 0
            # 8 Edge cases
            if i == 0 and j == 0:
                E = -J * s * (2*latt[i + 1, j] + 2*latt[i, j+1])
            elif i != n-1 and j == 0:
                E = -J * s * (latt[i + 1, j] + latt[i - 1, j] + latt[i, j + 1] + latt[i, j - 1])
            elif i == n-1 and j==0:
                E = -J * s * (2*latt[i - 1, j] + 2*latt[i, j + 1])
            elif i == 0 and j == n-1:
                E = -J * s * (2 * latt[i + 1, j] + 2*latt[i, j - 1])
            elif i == n-1 and j == n-1:
                E = -J * s * (2 * latt[i - 1, j] + 2 * latt[i, j - 1])
            elif i == 0:
                E = -J * s * (2*latt[i + 1, j] + latt[i, j + 1] + latt[i, j - 1])
            elif i == n-1:
                E = -J * s * (2 * latt[i - 1, j] + latt[i, j + 1] + latt[i, j - 1])
            elif j == n-1:
                E = -J * s * (latt[i + 1, j] + latt[i - 1, j] + 2 * latt[i, j - 1])
            else: assert(False)
        return E

    def get_E(self, latt=None):
        energies = [0, 0, 0, 0, 0]
        if type(latt) == type(None):
            latt = np.array(self.L)

        n = latt.shape[0]
        for i in range(n):
            for j in range(n):
                Eij = self.get_Eij(i,j,latt)
                if Eij == -4 * J:
                    energies[0] += 1
                elif Eij == -2 * J:
                    energies[1] += 1
                elif Eij == 0:
                    energies[2] += 1
                elif Eij == 2 * J:
                    energies[3] += 1
                elif Eij == 4 * J:
                    energies[4] += 1
                else:
                    print('anomalous energy', Eij)
                    assert (False)
        self.E = energies / np.sum(energies)
        return self.E

    def evolve(self, attempts=10):
        i, j = np.random.randint(0,n,size=2)
        Eij = self.get_Eij(i,j,self.L)

        if Eij >= 0:
            self.L[i,j] *= -1
            return True
        else:
            self.L[i,j] *= -1
            a = np.linalg.norm(self.get_E(latt=self.L) - self.pi)
            if a < self.pidist:
                self.pidist = a
                return True
            else:
                self.L[i,j] *= -1
        return False

sys = System()

running = True
i = 0
pygame.time.wait(1000)
while running:
    Poll_Events()

    if not sys.evolve():
        if i % 50 == 0:
            # print('P:', sys.E)
            pass
    if i % 50 == 0:
        redraw(sys.L)
        # print(sys.pidist, sys.E, sys.pi, np.sum(sys.L))
    i += 1