import numpy as np
import numpy.linalg as la


def Fij(ri, rj, G):
    rel_r = rj - ri
    return G * (1 / la.norm(rel_r, ord=2) ** 3) * rel_r


def rhs(t, state, masses, G):
    total_acc = np.zeros_like(state)

    for idx, (ri, mi) in enumerate(zip(state, masses)):
        for jdx, (rj, mj) in enumerate(zip(state[idx + 1 :], masses[idx + 1 :])):
            partial_force = Fij(ri[:3], rj[:3], G)
            total_acc[idx, 3:] += partial_force * mj
            total_acc[idx + jdx + 1, 3:] -= partial_force * mi

    total_acc[:, :3] = state[:, 3:]

    return total_acc


Msun = 1.98847 * 10 ** 30  ## Mass of the Sun, kg
AU = 149597871e3  ## 1 Astronomical Unit, m
year = 365.25 * 24 * 3600  ## 1 year, s
G = 4 * np.pi ** 2  ## in solar masses, AU, years
V = np.sqrt(G)


initial_state = np.array(
    [
        [0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.25, 0.9682458365518543, 0.0, 0.9682458365518543 * 0, -0.25 * 0, 0.0],
        [-0.5, -0.8660254037844386, 0.0, -0.8660254037844386 * 0, 0.5 * 0, 0.0],
    ]
)

masses = np.array(
    [
        1,
        1,
        1,
        1,
    ]
)

rhs(0.0, initial_state, masses, G)
