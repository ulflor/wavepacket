---
file_format: mystnb
kernelspec:
    name: python3
---

# Schroedinger cat states


This demo demonstrates one of the core features of Wavepacket:
Transition between wave functions and density operators with minimal fuss.

We consider a simple free particle as a coherent or incoherent sum of two Gaussians.


```{code-cell}
import math

import matplotlib.pyplot as plt
import numpy as np
import wavepacket as wp
```
