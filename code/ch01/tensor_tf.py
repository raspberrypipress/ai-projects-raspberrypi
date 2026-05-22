import tensorflow as tf
from PIL import Image

# Create a 0-dimensional (or rank 0) tensor
scalar = tf.constant(42)
print(scalar.shape) # Output: ()

# Create a 1D tensor (vector)
vector = tf.constant([1, 2, 3])
print(vector.shape) # Output: (3,)

# Creating a 2D tensor (matrix)
matrix = tf.constant([[1, 2], [3, 4]])
print(matrix.shape) # Output: (2, 2)

# Creating a 3D tensor (e.g., image with 3 colour channels)
# Produces a 128x128x3 tensor of values between 0 and 1
image_tensor = tf.random.uniform(shape=[128, 128, 3])
print(image_tensor.shape) # Output: (128, 128, 3)
image_array = tf.keras.utils.array_to_img(image_tensor)
print(image_array.shape)
image_array.show()

