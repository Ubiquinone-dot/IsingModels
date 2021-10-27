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
Na = 6.022e23   # mol^-1
kB = 1.3806e-23 # J/K
R = 8.314       # Na * kB

# Hyperparameters
J = 1     # coupling constant
T = 0.5

# Useful variables
b = 1/T
epsilon = 1e-10
if b > 1/epsilon:
    b = 0


print('Expected magnetisation for a 2-d lattice: x')
print('Expected average energy for the lattice (N * -J tanh(b*J)): {}'.format(-(25*25)*J*np.tanh(b*J)))

# End of pygame implementation


class System():
    # biased initialization, text box
    algorithm = 'MCMC (Metropolis) Sampling'

    cache_limit = int(1e+6)  # Maximum size of cache before reset
    lattice_area = 800
    n = 200  # n (columns)
    m = n   # m (rows)
    a = lattice_area//n # square side length
    p = 10 # padding

    pygame.init()
    pygame.display.set_caption('2D-Ising model | ' + algorithm + ' Implementation')
    pygame.display.set_icon(pygame.image.load('imgs/molecules.png'))
    font_size = 16
    font_local = pygame.font.SysFont('bahnschrift', font_size)
    bg_color = (25, 25, 25) # (250, 250, 250)
    ScreenWidth, ScreenHeight = ScreenDims = (2*p + lattice_area, 2*p + font_size + lattice_area)  # (1800, 920)
    win = pygame.display.set_mode(ScreenDims)
    colours = [(166, 166, 166), (0, 50, 50), (163, 0, 30)]  # [0] for pen, [1] for spin 1, [2] for spin -1
    counter_width = 75
    counter_bg_rect = pygame.Rect((ScreenWidth - counter_width, 0), (counter_width, p + font_size))
    def __init__(self, ):
        # System variables:
        self.n, self.m = self.size = (System.n, System.m) # n (columns) by m (rows) lattice
        self.N = self.n * self.m

        # Initialize lattice points, [1,1,-1,1,-1,...] randomly (for now)
        self.L = np.random.randint(0, 2, size=self.size) * 2 - 1

        # initialize variables for each parameter
        self.E0 = self.get_Ei()
        self.sE = 0         # running sum for each epoch
        self.Ebar = 0       # will be updated with each cache reset (end of each epoch).
        self.E = self.E0    # will be stored and changed by dE with each iteration to save compute
        self.latest_Ebar=0  # used for analysis only

        self.B0 = np.sum(self.L)/self.N
        self.sB = 0
        self.Bbar = 0
        self.B = self.B0

        # other variables
        self.epoch = 1
        p = System.p # padding
        a = System.a # side-length
        self.rects = [[pygame.Rect((p + i*a, p + System.font_size + j*a), (a,a)) for j in range(self.m)] for i in range(self.n)]

    def get_Ei(self):
        Ei = 0
        # Sum energies of 1 dimensional lattices spanning y, across x
        for i in range(self.n):
            for j in range(0,self.m-1):

                Ei += self.L[i,j] * self.L[i, j+1]
        # Sum energies of 1 dimensional lattices spanning x, across y
        for j in range(self.m):
            for i in range(0, self.n - 1):
                Ei += self.L[i, j] * self.L[i+1, j]

        return Ei * -J

    def get_dE(self, i, j): # Compute the corresponding change if spin i,j were to be flipped.
        if i == 0:
            si_prev, si_next = 0, self.L[i+1, j]
        elif i == self.n-1:
            si_prev, si_next = self.L[i-1, j], 0
        else:
            si_prev, si_next = self.L[i-1, j], self.L[i+1, j]

        if j == 0:
            sj_prev, sj_next = 0, self.L[i, j + 1]
        elif j == self.m - 1:
            sj_prev, sj_next = self.L[i, j - 1], 0
        else:
            sj_prev, sj_next = self.L[i, j - 1], self.L[i, j + 1]
        return 2 * J * self.L[i,j] * (si_prev + si_next + sj_prev + sj_next)

    def reset_cache(self):
        Ebari = self.sE / self.cache_limit  # retrieve average for current epoch
        self.latest_Ebar = Ebari
        self.Ebar = self.Ebar + (Ebari - self.Ebar) / self.epoch  # update running average

        Bbari = self.sB / self.cache_limit  # retrieve average for current epoch
        self.Bbar += (Bbari - self.Bbar) / self.epoch  # update running average

        self.sE = 0
        self.sB = 0

    def step(self):
        # Calculate properties wanted
        self.sE += self.E
        self.sB += self.B

        # Walk in space of states
        i, j = np.random.randint(0, self.n), np.random.randint(0, self.m)
        dE = self.get_dE(i, j)
        if dE <= 0:
            self.L[i, j] *= -1
            self.E += dE
        else:
            r = np.random.uniform(0, 1)
            if r <= np.exp(-b * dE):
                self.L[i, j] *= -1
                self.E += dE
                self.B += 2*self.L[i, j] / self.N

    def jump(self):
        # Calculate properties wanted
        self.sE += self.E
        self.sB += self.B

        # Walk in space of states
        c = 1
        mutot = 0
        i, j = np.random.randint(c+1, self.n-c-1), np.random.randint(c+1, self.m-c-1)

        # top and bottom part of square, sum over the interactions
        for x in range(-c, c+1):
            mutot += self.L[i+x,j-c] * self.L[i+x,j-c-1] + self.L[i+x,j+c] * self.L[i+x,j+c+1]
        # sides of square
        for y in range(-c, c+1):
            mutot += self.L[i-c-1,j+y] * self.L[i-c,j+y] + self.L[i+c,j+y] * self.L[i+c+1,j+y]
        dE = 2*J*mutot
        print(dE, np.exp(-b * dE), self.get_Ei())
        if dE <= 0:
            self.L[i-c:i+c,j-c:j+c] *= -1
            self.E += dE
            print('JUMPED <<<<',np.exp(-b * dE), self.get_Ei())
        else:
            r = np.random.uniform(0, 1)
            if r <= np.exp(-b * dE):
                self.L[i-c:i+c,j-c:j+c] *= -1
                self.E += dE
                self.B += 2*self.L[i, j] / self.N
                print('JUMPED <<<<',np.exp(-b * dE), self.get_Ei())

    def redraw(self):
        System.win.fill(System.bg_color)

        # draw spins as squares of sides 20 pixels
        for i in range(self.n):
            for j in range(self.m):
                pygame.draw.rect(System.win, System.colours[self.L[i, j]], self.rects[i][j])

        # draw data for system
        energy_counter = System.font_local.render('T = {}K | Lattice size = {} | Epoch: {} | <B> = {:.4f} | E/N={:.4f}, <E>/N={:.4f}'.format(
                                                                                                T,
                                                                                                self.size,
                                                                                                self.epoch,
                                                                                                self.Bbar,
                                                                                                self.E/self.N,
                                                                                                self.Ebar/self.N),
                                                                                                1, System.colours[0])

        System.win.blit(energy_counter, (5, 5))

        pygame.display.update()

    def blit_counter(self, iter):
        pygame.draw.rect(System.win, System.bg_color, System.counter_bg_rect)
        running_counter = System.font_local.render('{:.0f}/{:.0f}'.format(iter//1e4, System.cache_limit//1e4), 1, System.colours[0])
        System.win.blit(running_counter, (System.ScreenWidth - System.counter_width,5))
        pygame.display.update()