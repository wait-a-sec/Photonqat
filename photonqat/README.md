Library for photonic continuous variable quantum programming

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

F.Dgate(0, alpha) # Displacement to mode 0
F.Sgate(1, r) # Squeezeng to mode 1
```

### Plot Wigner function
```python
# Plot Wigner fucntion for mode 0 using matplotlib
(x, p, W) = F.Wignerfunc(0, plot = 'y', xrange = 5.0, prange = 5.0)
```

