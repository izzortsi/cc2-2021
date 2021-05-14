from explicit_runge_kutta.explicit_rk import (
    RK2G,
    RK3,
    RK4,
    RK4r38,
    RK3G,
    ForwardEuler,
)

Integrators = {
    "RK2G": RK2G,
    "RK3": RK3,
    "RK4": RK4,
    "RK4r38": RK4r38,
    "RK3G": RK3G,
    "ForwardEuler": ForwardEuler,
}
