from Systems.MCMC_2D import *

sys = System()

def Poll_Events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            global running
            running = False

epochs = 100000
System.cache_limit = int(1e+2)
running = True
while running:
    for epoch in range(1, epochs+1):
        sys.redraw()
        for iter in range(System.cache_limit):
            Poll_Events()
            sys.step()
            if iter % 1e4 == 0:
                sys.blit_counter(iter)

        sys.epoch = epoch
        sys.reset_cache()
        print('Cache reset... epoch:', epoch, 'Current average: <E>={:.8f} | <B>={:.8f} | Total steps: {} | E - Calculated E = {}'.format(
            sys.Ebar, sys.Bbar, epoch * System.cache_limit, sys.E - sys.get_Ei()))
    running = False

print('Average magnetisation:', sys.sB / iter)
print('Expected magnetisation for a 1-d lattice: 0')
print('Average Energy per particle:', sys.Ebar / sys.N)
print('-Jtanh(Jb): {}'.format(-J*np.tanh(b*J)))

pygame.quit()