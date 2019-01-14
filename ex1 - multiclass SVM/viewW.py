import numpy as np
import matplotlib.pyplot as plt

W = np.load('W.npy')

fig = plt.figure()
for i in range(0, 4):
    plt.subplot(2, 2, i+1)
    plt.imshow(np.reshape(W[i], (28, 28)))
    plt.clim((-0.10, 0.10))
    plt.colorbar()
    plt.title(str(i))
    plt.axis('off')

plt.savefig('view of W', dpi=fig.dpi, bbox_inches='tight')
plt.show()
