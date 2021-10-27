# import matplotlib.pyplot as plt
import numpy as np
import pygame

'''
This file implements the 1d ising model.

algorithm outline:
Compute the expectation value of a property A, across the space of states with a boltzmann distribution:
<A> = sum(ith state) P(Ei) * Ai approximates to 1/M * sum(kth sample) Ak
by sampling the space using a markov chain with transition probabilities following Pi->k = min(1,exp(-beta * (Ek - Ei)))
Thus for a single transition (step in the algorithm):
Initialize to state i
for iter:
    compute the change in energy Ek - Ei,
    log Ai or add to sum
    if dE < 0: transition to state k
    else:
        # probability of transition is exp(-beta*dE) (which ranges from (0 -> 1])
        select random uniform r (0,1] and transition if r < exp(-beta*dE)

'''

# Scientific constants
Na = 6.022e23
kB = 1.3806e-23 # J/K
R = 8.314 # Na * kB

# Hyperparameters
n = 830     # number of lattice sites
J = 1     # coupling constant
T = 1000

# Useful variables
b = 1/T
epsilon = 1e-7


print('Expected magnetisation for a 1-d lattice: 0')
print('Expected average energy for the lattice (N * -J tanh(b*J)): {}'.format(-n*J*np.tanh(b*J)))

# End of pygame implementation


class System():
    # biased initialization, text box
    algorithm = '1D MCMC Sampling'

    cache_limit = int(1e+6)  # Maximum size of cache before reset

    pygame.init()
    pygame.display.set_caption('1D-Ising model | ' + algorithm + ' Implementation')
    pygame.display.set_icon(pygame.image.load('imgs/molecules.png'))
    font_local = pygame.font.SysFont('bahnschrift', 16)
    bg_color = (25, 25, 25) # (250, 250, 250)
    ScreenWidth, ScreenHeight = ScreenDims = (1100, 650)  # (1800, 920)
    win = pygame.display.set_mode(ScreenDims)
    spin_colours = [0, (0, 50, 50), (163, 0, 30)]  # draw lattice points with -1 spin grey, +1 as cyan

    def __init__(self):
        # System variables:

        # Initialize lattice points, [1,1,-1,1,-1,...] randomly (for now)
        self.L = np.random.randint(0, 2, size=n) * 2 - 1

        # initialize variables for each parameter
        self.E0 = self.get_Ei()
        self.sE = 0         # running sum for each epoch
        self.Ebar = 0       # will be updated with each cache reset (end of each epoch).
        self.E = self.E0    # will be stored and changed by dE with each iteration to save compute

        self.sB = 0
        self.Bbar = 0

        # other variables
        self.epoch = 0
        self.Rects = []
        for i, spin in enumerate(self.L):
            a = 20
            r = 10
            w = int((r + 2 * i * r) / System.ScreenWidth)
            nw = (System.ScreenWidth - 2 * r) / (2 * r)
            x = r + 2 * r * (i - w * nw - w)
            y = 4 * r * (1 + w)
            self.Rects.append(pygame.Rect((x,y), (a,a)))

    def get_Ei(self):
        Ei = 0
        for i, spin in enumerate(self.L[:-1]):
            Ei += spin * self.L[i+1]
        return Ei * -J

    def get_dE(self, i): # Compute the corresponding change if spin i were to be flipped.
        if i == 0:
            s_prev, s_next = 0, self.L[i+1]
        elif i == n-1:
            s_prev, s_next = self.L[i-1], 0
        else:
            s_prev, s_next = self.L[i-1], self.L[i+1]
        return 2 * J * self.L[i] * (s_next + s_prev)

    def reset_cache(self, epoch, cache_size):
        Ebari = self.sE / cache_size # retrieve average for current epoch
        self.Ebar = self.Ebar + (Ebari - self.Ebar) / epoch # update running average

        self.sE = 0

    def step(self):
        # Calculate properties wanted
        self.sE += self.E
        self.sB += 0  # np.sum(self.L)/n

        # Walk in space of states
        i = np.random.randint(0,n)
        dE = self.get_dE(i)
        if dE >= 0:
            self.L[i] *= -1
            self.E += dE
        else:
            r = np.random.uniform(0,1)
            if r <= np.exp(-b * dE):
                self.L[i] *= -1
                self.E += dE

    def redraw(self):
        System.win.fill(System.bg_color)

        # draw spins as squares of sides 20 pixels
        for i, (spin, rect) in enumerate(zip(self.L, self.Rects)):
            pygame.draw.rect(System.win, System.spin_colours[spin], rect)

        # draw data for system
        energy_counter = System.font_local.render('<E> = {:.4f}  |  Epoch: {}'.format(self.Ebar, self.epoch), 1, (0, 60, 0))

        System.win.blit(energy_counter, (5, 5))

        pygame.display.update()
