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

algorithm = '1D MCMC Sampling'

# Scientific constants
Na = 6.022e23
kB = 1.3806e-23
R = 8.314 # Na * kB

# Hyperparameters
n = 830     # number of lattice sites
J = 1       # coupling constant
T = 300

# Useful variables
b = 1/T
epsilon = 1e-7

## Start of pygame implementation
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

### End of pygame implementation

class System():
    cache_limit = 1000000 # Number of iterations before cache reset
    def __init__(self):
        # Initialize lattice points, [1,1,-1,1,-1,...] randomly
        self.L = np.random.randint(0, 2, size=n) * 2 - 1
        # initialize variables for each parameter
        self.E0 = self.get_Ei()
        self.sE = 0 # running sum
        self.Ebar = 0 # will be updated with each cache reset.
        self.E = self.E0 # will be stored and updated with each iteration to save compute

        self.sB = 0
        self.Bbar = 0


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
        return 2 * J * (s_next + s_prev)

    def reset_cache(self):
        pass


    def step(self):
        # Calculate properties wanted
        B = np.sum(self.L)/n
        self.sB += B
        E = self.get_Ei()
        self.sE += E

        # Walk in space of states
        i = np.random.randint(0,n)
        dE = self.get_dE(i)
        if dE > 0:  self.L[i] *= -1
        else:
            r = np.random.uniform(0,1)
            if r < np.exp(-b * dE):  self.L[i] *= -1

sys = System()

running = True
iterations, iter = int(1E+10), 0
while iter < iterations and running:
    Poll_Events()
    sys.step()


    if iter % 1000 == 0: redraw(sys.L)
    iter+=1

print('Terminated at iteration {} of {} ({}% complete)'.format(iter, iterations, 100*iter/iterations))
print('Average magnetisation:', sys.sB / iter)
print('Expected magnetisation for a 1-d lattice: 0')
print('Average Energy:', sys.sE / iter)
print('Expected average energy for the lattice (N * -J tanh(b*J)): {}'.format(-n*J*np.tanh(b*J)))