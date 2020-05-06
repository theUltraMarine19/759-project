import matplotlib.pyplot as plt
import numpy as np

x = np.loadtxt("output.txt")

print(x.shape)

plt.imshow(x)
plt.axis('off')
# plt.show()
plt.savefig("results.jpg")
print("Results output to results.jpg")
