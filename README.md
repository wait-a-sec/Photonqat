Library for photonic continuous variable quantum programming

![Logo]](https://github.com/ryuNagai/Photonqat/blob/master/Logo.jpg?raw=true)


# Install
pip install photonqat

# Circuit

## Fock basis

### Circuit
```python
import photonqat as pq
import numpy as np
import matplotlib.pyplot as plt

# mode number = 2, cutoff dimension = 15
F = pq.Fock(2, cutoff = 15)
```

### Applying gate
```python
alpha = (1 + 1j)
r = -0.5

F.D(0, alpha) # Displacement to mode 0
F.S(1, r) # Squeezeng to mode 1
```

### Plot Wigner function
```python
# Plot Wigner fucntion for mode 0 using matplotlib
(x, p, W) = F.Wigner(0, plot = 'y', xrange = 5.0, prange = 5.0)
```

## Gaussian formula

### Circuit
```python
import photonqat as pq
import numpy as np
import matplotlib.pyplot as plt

# mode number = 2
G = pq.Gaussian(2)
```
Applying gate and plotting Wigner function are also available in same fasion as Fock basis.
But there are differences in availavle getes and measurement.

Code examples are [here](https://github.com/ryuNagai/Photonqat/tree/master/examples).