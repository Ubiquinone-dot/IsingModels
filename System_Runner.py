from Systems.MCMC_2D import *

sys = System()

def Poll_Events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            global running
            running = False

Iterations = int(1e+8)
System.cache_limit = int(1e+3)
epochs = Iterations // System.cache_limit
print('Epochs: {}, Cache limit: {}, Iterations: {}'.format(epochs, System.cache_limit, Iterations))
counter = False
pygame.time.wait(500)
running = True
while running:
    for epoch in range(1, epochs+1):
        if epoch % 25 == 0:
            sys.jump()
        sys.redraw()
        for iter in range(System.cache_limit):
            Poll_Events()
            sys.step()
            if counter and iter % 1e4 == 0:
                sys.blit_counter(iter)

        sys.epoch = epoch
        sys.reset_cache()
        #print('Cache reset... epoch:', epoch, 'Current average: <E>={:.8f} | <B>={:.8f} | Total steps: {} | E - Calculated E = {}'.format(
        #    sys.Ebar, sys.Bbar, epoch * System.cache_limit, sys.E - sys.get_Ei()))

    sys.redraw()

print('Average magnetisation:', sys.sB / iter)
print('Expected magnetisation for a 1-d lattice: 0')
print('Energies:', sys.Ebar, sys.Ebar)
print('-Jtanh(Jb): {}'.format(-J*np.tanh(b*J)))

pygame.quit()