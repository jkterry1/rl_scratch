import numpy as np
from array2gif import write_gif

arrays = []
for i in range(125):
    rand_array = np.random.randint(0, 255, (880, 560))
    stack = np.stack([rand_array, np.zeros(880, 560), np.zeros(880, 560)], axis=0)
    arrays.append(stack)

write_gif(arrays, 'test.py', fps=15)