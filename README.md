# symbolic_lp
Toolkit to express linear program constraints in a symbolic and natural way and get parameterized matrix constraints. 

## Installation
pip install symbolic-lp

## Demo

```python
import numpy as np
import torch
from symbolic_lp import Model

# Create a model
model = Model()

# Add variables and parameters
x = model.add_var("x", 2)
p = model.add_param("p", 2)

# Add constraints
model.add_constraint(x[0] + p[0] <= 10)
model.add_constraint(2 * x[1] - p[1] >= 5)

# Build matrices for specific parameter values
A, b = model.build_matrices(param_values={"p": [3, 4]})
print("A:", A)
print("b:", b)
```

 ## Demo with batched parameter values
``` python
np_batched_p_values = np.random.randn(20, 2) + 4
torch_batched_p_values = torch.tensor(np_batched_p_values, dtype=torch.float32)

for batched_p_values in [np_batched_p_values, torch_batched_p_values]:
    A_batched, b_batched = model(param_values={"p": batched_p_values})
    print("A_batched shape:", A_batched.shape)
    for i in range(20):
        assert np.allclose(A_batched[i], [[1, 0], [0, -2]])
        assert np.allclose(b_batched[i], [10 - (batched_p_values[i][0]), -5 - (batched_p_values[i][1])])
```