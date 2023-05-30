import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x = np.random.rand(10, 10)
y = pd.DataFrame(x)

xpoints = np.array([0, 6])
ypoints = np.array([0, 250])
print(xpoints)
print(ypoints)

plt.plot(xpoints, ypoints)
plt.show()

plt.plot(x)
plt.show()