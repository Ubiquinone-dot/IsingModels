'''
2d Ising model using pseudo monte-carlo simulation with adjacent spin approximation

Algorithm:
Initialize a 2d grid of random +1/-1 spins
Pick a random point on the grid
approximate energy as energy due to interaction with 4 nearest neighbours
if E > 0:   flip spin
elif E < 0:
    r = [0,1)
    if r < e^2E/T: flip spin


m = sum(spins) is monitored for each epoch
'''

# Pyramidal import statements
import matplotlib.pyplot as plt
import numpy as np

lw = 20
J = 1
T = 1
b = 1/T
kb = 1
grid_size = (lw,lw)
grid = np.random.randint(2, size=grid_size)
grid[grid < 1] = -1


i,j = np.random.randint(0,lw, size=(2))
i, j = (0, 0)
s = grid[i,j]
s_n, s_s, s_e, s_w = grid[i,j-1], grid[i,(j+1) * (j+1 < lw)], grid[(i+1) * (i+1 < lw),j], grid[i-1,j]
s_sur = s_n + s_s + s_e + s_w

E = - J * b * s * (s_sur)
if E > 0:
    grid[i,j] *= -1
elif E < 0:
    r = np.random.uniform(0,1)
    p = np.exp(- E / (T * kb))
    print(p, r)
    if r < p: grid[i,j] *= -1

print(grid, E, s_sur, (i,j))



# here's the pseudo monte-carlo attempt:
'''
# pick random index within 2x2 unit cell
index = np.random.randint(1,5)
print(index)
col_i = [col for col in range((index % 2), grid_size[0], 2)]
row_i = [row for row in range((index % 2), grid_size[1], 2)]
print(row_i, col_i)


print()
print(grid)
'''

# The code for finite 2d non-spherical surface
'''
except IndexError:
    n_neighbours = 4
    s_sur = 0
    for indices in ([i,j-1], [i,j+1],[i+1,j], [i-1,j]):
        try:
           s_sur += grid[indices[0], indices[1]]
        except IndexError:
            n_neighbours -= 1
   '''