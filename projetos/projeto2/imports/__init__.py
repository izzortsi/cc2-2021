import os
import sys
import numpy as np
import numpy.linalg as la
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from explicit_runge_kutta.explicit_rk import ExplicitRungeKutta
from explicit_runge_kutta.integrators import Integrators

KURAMOTO_OUTS = os.path.join("kuramoto", "kuramoto_outputs")
if not os.path.exists(KURAMOTO_OUTS):
    os.mkdir(KURAMOTO_OUTS)

THREEBODY_OUTS = "3body_outputs"
if not os.path.exists(THREEBODY_OUTS):
    os.mkdir(THREEBODY_OUTS)

CHOREOGRAPHIES = {
    "2": {"v": np.array([0.322184765624991, 0.647989160156249]), "T": 51.3958},
    "3": {"v": np.array([0.257841699218752, 0.687880761718747]), "T": 55.6431},
    "4": {"v": np.array([0.568991007042164, 0.449428951346711]), "T": 51.9645},
    "22": {"v": np.array([0.698073236083981, 0.328500769042967]), "T": 100.846},
}
