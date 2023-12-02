import numpy as np
import matplotlib.pyplot as plt



def plot(t, y, title):
    plt.plot(t, y, label='Approximate Solution')
    plt.xlabel('Time')
    plt.ylabel('y(t)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()