import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.array([1, 2, 3, 4, 5])
s = np.array([89.74, 89.78, 89.81, 89.79, 89.79])

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='Number of sub-pillars', ylabel='Average precision (%)',
       title='Number of sub-pillars VS. Average precision (AP)')
ax.grid()

fig.savefig("test.png")
plt.show()