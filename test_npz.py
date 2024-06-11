import numpy as np

np.set_printoptions(threshold=np.inf)

# Load the npz file
data = np.load('path to sensor_response.npz/impulse_response.npz')

# Access and print each array in the npz file
for key in data.files:

    print("key is ", key)
    print(data[key])

data.close()


