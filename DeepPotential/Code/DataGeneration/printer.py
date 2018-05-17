
import numpy as np
import time


class ProgressTimer:
    
    def __init__(self, iterations, print_every=1):
        self.start_time = time.time()
        self.iterations = iterations
        self.iteration = 0
        self.print_every = print_every
    
    def how_long(self):
        self.iteration += 1
        if self.iteration % self.print_every == 0:
            print(np.round(100*self.iteration/self.iterations, 1), ' % --- ', 
                  np.round((time.time()-self.start_time)/
                           self.iteration*(self.iterations-self.iteration)/60,
                           1))