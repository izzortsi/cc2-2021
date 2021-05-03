from numpy import *

h = 0.0001
t = arange(0, 100 * pi, h)
y = [1, 0]
k = zeros((2, 4))
error = 0
t = 0
while t < 100 * pi:
    k[0, 0] = h * y[1]
    k[1, 0] = -h * y[0]
    k[0, 1] = h * (y[1] + k[1, 0] / 2)
    k[1, 1] = -h * (y[0] + k[0, 0] / 2)
    k[0, 2] = h * (y[1] + k[1, 1] / 2)
    k[1, 2] = -h * (y[0] + k[0, 1] / 2)
    k[0, 3] = h * (y[1] + k[1, 2])
    k[1, 3] = -h * (y[0] + k[0, 2])
    y[0] = y[0] + k[0, 0] / 6 + k[0, 1] / 3 + k[0, 2] / 3 + k[0, 3] / 6
    y[1] = y[1] + k[1, 0] / 6 + k[1, 1] / 3 + k[1, 2] / 3 + k[1, 3] / 6
    t += h
    error += fabs(cos(t) - y[0])
print("4th order Runge Kutta with step size = " + str(h))
print("Average error = " + str(error / ((int)(100 * pi / h))))
print("Final value error = " + str(cos(t) - y[0]))
print("Final slope error = " + str(sin(t) - y[1]))
