from matplotlib import animation as animation
from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np

# System Settings
# Example settings which are interesting:
# [0,0,1,1] -
# Coherent = True (eigenstate of annihilation operator)
coherent = False
states = [0,0,0,0,0,0,0,0,0,0,1]
states = [0,1]

if coherent:
    states = [1,1,1,1,1,1,1,1,1,1,1]
    for n in range(1, len(states)):
        states[n] = 1/np.math.factorial(n)**0.5

resolution = 150
xspacing = 7 # range from -x to x
interval = 10
DX = 2*xspacing/resolution
xs = np.linspace(-xspacing, xspacing, resolution).astype(float)
PRINT = True

style.use('fivethirtyeight')

# Making axes and figure
fig = plt.figure()
plt.axis('off')
fig.patch.set_facecolor('#252525')
axis1 = fig.add_subplot(1,2,1)
axis2 = fig.add_subplot(1,2,2)
titletext = '|Ψ> ∝ '
for n, nweight in enumerate(states):
    if nweight != 0:
        if nweight == 1:
            titletext += '|'+str(n)+'> '
        elif nweight == 1j:
            titletext += 'j|'+str(n)+'>'
        else:
            titletext += str(nweight)+'|'+str(n)+'> '

        if n != len(states)-1:
            titletext += '+ '
# axis1.set(xlim=(-xspacing, xspacing),ylim=(-1,1))

# Constants
hbar = 1

# Normalises coeffs
coeffs = np.array(states).astype(complex)
coeffs = coeffs / np.sum(coeffs * coeffs.conjugate()).real**0.5


def Hermite(x, n): # hardcoded cos im lazy
    if n == 0: return 1
    elif n == 1: return 2 * x
    elif n == 2: return 4 * x**2 - 2
    elif n == 3: return 8 * x**3 - 12 * x
    elif n == 4: return 16 * x**4 - 48 * x**2 + 12
    elif n == 5: return 32 * x**5 - 160 * x**3 + 120 * x
    elif n == 6: return 64 * x**6 - 480 * x**4 + 720 * x**2 - 120
    elif n == 7: return 128 * x**7 - 1344 * x**5 + 3360 * x**3 - 1680 * x
    elif n == 8: return 256 * x**8 - 3584 * x**6 + 13440 * x**4 - 13440 * x**2 + 1680
    elif n == 9: return 512 * x**9 - 9216 * x**7 + 48384 * x**5 - 80640 * x**3 + 30240 * x
    elif n == 10: return 1024 * x**10 - 23040 * x**8 + 161280 * x**6 - 403200 * x**4 + 302400 * x**2 - 30240
    else:
        print('Hermite not coded to n>5')
        assert(False)


class State:
    def __init__(self, coeffs=[1]):
        self.coefficients = coeffs
        p=0
        for c in coeffs:
            p += c*c.conjugate()
        assert(0.99 < p < 1.01)
        self.mass = 1
        self.omega = 1
        self.maxn = len(coeffs) - 1
        self.alpha = self.mass * self.omega / hbar
        self.GStateMapping = {} # dict of energy level to ground state mapping of y to x in linspace specified
        self.CreateGStateMapping()
        self.CStateMapping = self.GStateMapping


    def Nn(self, n): # returns normalisation constant for energy level
        return pow(2**(n) * np.math.factorial(n), -0.5) * pow(self.alpha / 3.14, 0.25)

    def exp(self, x):
        return pow(2.72, -x**2 * self.alpha/2)

    def CreateGStateMapping(self):
        global resolution, xspacing

        # Groundstate mappings
        xs = np.linspace(-xspacing, xspacing, resolution)
        for n in range(self.maxn+1):
            ys = []
            Nn = self.Nn(n)
            for x in xs:
                y = Nn * self.exp(x) * Hermite(pow(self.alpha,0.5)*x, n) + 0j
                ys.append(y)

            self.GStateMapping[str(n)] = np.array(ys)

    def Superposed_ys(self):
        ys = self.coefficients[self.maxn] * self.CStateMapping[str(self.maxn)]

        for n in range(self.maxn):

            ys += self.CStateMapping[str(n)] * self.coefficients[n]
        return ys

    def Evolve_System(self, t):
        for n in range(self.maxn+1):
            if self.coefficients[n] != 0:
                z = 0 + 1j
                z *= np.sin(self.omega * t * (n + 0.5))
                z += np.cos(self.omega * t * (n + 0.5))

                # self.CStateMapping[str(n)] = np.multiply(self.CStateMapping[str(n)], z)
                self.CStateMapping[str(n)] *= z


# Example Superposed states:
# [1/3**0.5, 1/3**0.5, 1/3**0.5]
# [1/5**0.5,-1/5**0.5,1/5**0.5,1/5**0.5,1/5**0.5]
# [1]
Psi = State(coeffs)
def evolve(i):

    #plot on axis 1
    ys = Psi.Superposed_ys()
    axis1.clear()
    axis1.plot(xs, ys.real, color='#02b01c', label='Re(Ψ)')
    axis1.plot(xs, ys.imag, color='#007e9e', label='Im(Ψ)', )
    axis1.legend()
    #axis1.axis('off')
    axis1.set_facecolor('#303030')
    axis1.set_ylim((-1,1))
    Psi.Evolve_System(interval*0.01)
    axis1.set_title(titletext, fontsize=40, loc='left')
    axis1.xaxis.set_label_position('top')
    axis1.set_xlabel(' ')

    # plot psi^2 on axis 2
    axis2.clear()
    axis2.plot(xs, (ys * ys.conjugate()).real, color='#0070a3', label='ΨΨ*')
    axis2.set_ylim((-1,1))
    axis2.set_facecolor('#303030')
    axis2.legend()
    #axis2.axis('off')

    if PRINT and i % 3 == 0:
        # The expectation value of position
        Xbar = np.sum(np.multiply(xs, DX *(ys * ys.conjugate()).real))

        # Normalisation Check
        space_integrals = []
        for n in range(Psi.maxn+1):
            space_integrals.append(np.sum(DX * (ys * ys.conjugate()).real))


        # Orthogonality Check
        overlap = np.sum(np.multiply(Psi.CStateMapping['0'].conjugate(), Psi.CStateMapping['1']))
        print('Xbar =',Xbar)
        print('Integral over space of each eigenstate:',space_integrals)
        print('overlap:', overlap)

        # Energy of system
        energy = 0
        for n, alpha in enumerate(states):
            energy += alpha * Psi.omega * hbar * (n + 1/2)

        print(energy)


ani = animation.FuncAnimation(fig, evolve, interval = interval)
plt.show()

