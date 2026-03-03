"""
Baseline fast-core model
Meta-Stable Architectures Trilogy
Computational exploration repository

This file contains the minimal deterministic fast-layer model
described in Volume I–III.
"""

import numpy as np


def fast_core(x, y, k, lam, alpha, mu, gamma):
    """
    Fast core dynamical system.

    dx/dt = -k x + lam y
    dy/dt = alpha x + mu y - gamma y^3
    """

    dx = -k * x + lam * y
    dy = alpha * x + mu * y - gamma * y**3

    return dx, dy
  
  meta-stable-architectures/
│
├── README.md
└── simulations/
    ├── README.md
    └── baseline_model.py
