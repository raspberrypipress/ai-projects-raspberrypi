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
image_tensor = tf.random.uniform(shape=[64, 64, 3])
print(image_tensor.shape) # Output: (64, 64, 3)

image_array = tf.keras.utils.array_to_img(image_tensor)
image_array.show()
