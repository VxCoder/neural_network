import struct
import numpy as np
import matplotlib.pyplot as plt

with open("test_data", 'rb') as imgpath:
    magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
    images = np.fromfile(imgpath, dtype=np.uint8).reshape(num, 784)


fig, ax = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    sharey=True, )

ax = ax.flatten()
img = images[0].reshape(28, 28)
ax[0].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
