import numpy as np
from array2gif import write_gif

arrays = []
for i in range(125):
    arrays.append(np.random.randint(0, 255, (3, 880, 560)))

write_gif(arrays, 'test.py', fps=15)