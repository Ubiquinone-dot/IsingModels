import matplotlib.pyplot as plt
import numpy as np
import pygame

'''
This file implements the 1d ising model, with the sole focus on generating a state which obeys the boltzmann distribution
This is an initial draft algorithm which is not meant to be computationally efficient

algorithm outline:
Calculate the distribution of energies of the entire system.
The desired state is such that the relative amounts of each state is proportional to exp(-beta * Ei)
For the 1d ising model, there are only 3 energies associated with each spin: 2J, 0, -2J

'''

algorithm = 'Brute Force'

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
    for i, spin in enumerate(lattice):
        r=10
        w = int((r+2*i*r)/ScreenWidth)
        nw = (ScreenWidth - 2*r)/(2*r)
        x = r+2*r*(i-w*nw-w)
        pygame.draw.circle(win, colours[spin], (int(x), 4*r*(1+w)), r)
    pygame.display.update()

### PYGAME IMPLEMENTATION

n = 1000
epsilon = 1e-7
J = 1
T = 0.01
b = 1/T
lattice = np.random.randint(0, 2, size=(n)) * 2 - 1

E = {
    '2J':0,
    '0':0,
    '-2J':0
}


class System():
    def __init__(self):
        # Initialize lattice points, [1,1,-1,1,-1,...] randomly
        self.L = np.random.randint(0, 2, size=(n)) * 2 - 1

        t = [np.exp(b * 2 * J), 2 * np.exp(-b * 0), np.exp(-b * 2 * J)]
        self.pi = t / np.sum(t)
        print('pi:', self.pi)
        self.pidist = 1/epsilon
        self.get_E()

    def get_E(self, latt=None):
        energies = [0, 0, 0]
        if type(latt) == type(None):
            latt = np.array(self.L)

        for i, point in enumerate(latt):
            if i == 0:
                E = -J * point * (2 * latt[i + 1])
            elif i == n - 1:
                E = -J * point * (2 * latt[i - 1])
            else:
                E = -J * point * (latt[i + 1] + latt[i - 1])

            if E < 0:
                energies[0] += 1
            elif E > 0:
                energies[2] += 1
            else:
                energies[1] += 1
        self.E = energies / np.sum(energies)
        return self.E

    def evolve(self, iter=10):
        for i in np.random.randint(0,n, size=iter):
            self.L[i] *= -1

            a = np.linalg.norm(self.get_E(latt=self.L) - self.pi)
            if a < self.pidist:
                self.pidist = a
                return True
            else:
                self.L[i] *= -1
        return False


sys = System()

running = True
i = 0
pygame.time.wait(1000)
while running:
    Poll_Events()

    if not sys.evolve():
        if i % 50 == 0:
            print('P:', sys.E)
    if i % 50 == 0:
        redraw(sys.L)
        # print(sys.pidist, sys.E, sys.pi, np.sum(sys.L))
    i += 1