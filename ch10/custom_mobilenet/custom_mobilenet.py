import os  # models/ directory, list converted outputs, assert packerOut.zip exists
import subprocess  # optional nvidia-smi probe; imxconv-tf conversion subprocess
from typing import Generator  # type hints for representative-dataset generator

import keras  # losses/metrics for quantized_model.compile
import matplotlib.pyplot as plt  # visualize_detection (imshow, labels)
import model_compression_toolkit as mct  # IMX500 PTQ, TPC, export quantized Keras model
import numpy as np  # batch dims and arg-sorting predictions in visualization helpers
import tensorflow as tf  # data pipeline, augmentation, training, optimizers, callbacks
import tensorflow_datasets as tfds  # load rock_paper_scissors train/test splits
from keras.applications import MobileNetV2  # ImageNet backbone for transfer learning
from keras.layers import Dense, GlobalAveragePooling2D  # classification head on frozen base
from keras.models import Model  # functional API model wrapping base + head
from model_compression_toolkit.core import QuantizationErrorMethod  # MSE option in QuantizationConfig

# Model directories
MODELS_DIR = 'models/'  # directory for saved models and converted outputs
if not os.path.exists(MODELS_DIR):  # check if directory exists
    os.mkdir(MODELS_DIR)  # create directory if it doesn't exist

MODEL = MODELS_DIR + 'mobilenet-rps'  # full path to saved mode l
MODEL_KERAS = MODELS_DIR + 'mobilenet-quant-rps.keras'  # full path to quantized Keras model

BATCH_SIZE = 32  # batch size for training and validation
IMAGE_SHAPE = (224, 224)  # input shape for MobileNetV2 (height, width)

"""# Dataset
We will use the RPS dataset from [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors)
"""

# Load the RPS dataset
dataset_name = "rock_paper_scissors"
(train_ds, validation_ds), info = tfds.load(
    name=dataset_name,
    split=["train", "test"],
    with_info=True,
    as_supervised=True,
    shuffle_files=True,
    batch_size=BATCH_SIZE
)

# Display dataset information
print(info)

class_names = info.features['label'].names
num_classes = len(class_names)
print("class_names:", class_names)

"""## Pre-processing and augmentation"""

def preprocess_data(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SHAPE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

train_ds = train_ds.map(preprocess_data)
validation_ds = validation_ds.map(preprocess_data)

# Add augmentation
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
validation_ds = validation_ds.prefetch(tf.data.experimental.AUTOTUNE)

"""# Keras Mobilenetv2 model for transfer learning"""

base_model =  MobileNetV2(weights='imagenet', include_top=False, input_shape=IMAGE_SHAPE+(3,))

# Add custom classification layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, activation=tf.nn.softmax)(x)

# Create the full model
float_model = Model(inputs=base_model.input, outputs=predictions)

# Freeze layers in the base model
for layer in base_model.layers:
    layer.trainable = False

float_model.summary()
"""
Top and buttom layers.
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to
==================================================================================================
 input_1 (InputLayer)        [(None, 224, 224, 3)]        0         []

 Conv1 (Conv2D)              (None, 112, 112, 32)         864       ['input_1[0][0]']
 ...
 global_average_pooling2d (  (None, 1280)                 0         ['out_relu[0][0]']
 GlobalAveragePooling2D)

 dense (Dense)               (None, 3)                    3843      ['global_average_pooling2d[0][
                                                                    0]']

==================================================================================================
Total params: 2261827 (8.63 MB)
Trainable params: 3843 (15.01 KB)
Non-trainable params: 2257984 (8.61 MB)
__________________________________________________________________________________________________
"""

"""# Train"""

EPOCHS = 20

float_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    baseline=0.8,
    min_delta=0.01,
    mode='max',
    patience=5,
    verbose=1,
    restore_best_weights=True,
    start_from_epoch=5,
)

history = float_model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    callbacks=[callback]
)

float_model.save(MODEL)

"""# Quantization
For full integer quantization, you need to calibrate or estimate the range (Min, Max) of all floating-point tensors in the model. Unlike constant tensors such as weights and biases, variable tensors such as model input, activations (outputs of intermediate layers), and model output cannot be calibrated unless we run a few inference cycles. For this, the converter requires a representative dataset to calibrate with. This dataset can be a small subset (around 100-500 samples) of the training or validation data.

For details, see [Post-Training Quantization Example of MobileNetV2 Keras Model](https://github.com/sony/model_optimization/blob/v2.0.0/tutorials/notebooks/keras/ptq/example_keras_imagenet.ipynb)

Observe that we are using training part of the dataset as representative dataset.


"""

n_iter=10

# Create representative dataset generator
def get_representative_dataset() -> Generator:
    """A function that loads the dataset and returns a representative dataset generator.

    Returns:
        Generator: A generator yielding batches of preprocessed images.
    """
    dataset = train_ds

    def representative_dataset() -> Generator:
        """A generator function that yields batch of preprocessed images.

        Yields:
            A batch of preprocessed images.
        """
        for _ in range(n_iter):
            yield dataset.take(1).get_single_element()[0].numpy()

    return representative_dataset

# Create a representative dataset generator
representative_dataset_gen = get_representative_dataset()

# Specify the IMX500-v1 target platform capability (TPC)
tpc = mct.get_target_platform_capabilities("tensorflow", 'imx500', target_platform_version='v1')

# Set the following quantization configurations:
# Choose the desired QuantizationErrorMethod for the quantization parameters search.
# Enable weights bias correction induced by quantization.
# Enable shift negative corrections for improving 'signed' non-linear functions quantization (such as swish, prelu, etc.)
# Set the threshold to filter outliers with z_score of 16.
q_config = mct.core.QuantizationConfig(activation_error_method=QuantizationErrorMethod.MSE,
                                       weights_error_method=QuantizationErrorMethod.MSE,
                                       weights_bias_correction=True,
                                       shift_negative_activation_correction=True,
                                       z_threshold=16)

ptq_config = mct.core.CoreConfig(quantization_config=q_config)

quantized_model, quantization_info = mct.ptq.keras_post_training_quantization(
    in_model=float_model,
    representative_data_gen=representative_dataset_gen,
    core_config=ptq_config,
    target_platform_capabilities=tpc)

# Export the quantized model
mct.exporter.keras_export_model(model=quantized_model, save_model_path=MODEL_KERAS)

"""# Evaluation"""

float_model.evaluate(validation_ds)

quantized_model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
quantized_model.evaluate(validation_ds)

"""# Visualize detections"""
# # Disabled as we are running in headless docker container

# Load the test part of the dataset
test_ds, info = tfds.load(dataset_name, split=["test"], with_info=True)
print(info)

# Preprocess the input image for inference
def preprocess_image_visualization(image):
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = tf.image.resize(image, (224, 224))
    return image

# Perform detection on the input image
def detect_objects(model, image):
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    return predictions

# Get the class label and confidence score of the detected objects
def get_top_prediction(predictions):
    top_idx = np.argsort(predictions)[0][-1]
    top_score = predictions[0][top_idx]
    top_class = class_names[top_idx]
    return top_class, top_score

# Visualize the detections
def visualize_detection(image, cls, score):
    plt.imshow(image)
    plt.text(10, 20, f'{cls}: {score}', color='red')
    plt.axis('off')
    # plt.show() # disabled as we are in headless docker

# Visualize detection results for some images
# for sample in test_ds[0].take(4):
#     image = preprocess_image_visualization(sample['image'])
#     predictions = detect_objects(quantized_model, image)  # float_model/quantized_model
#     cls, score = get_top_prediction(predictions)
#     print(f'cls: {cls}, score: {score}')
#     assert score > 0.55
#     visualize_detection(sample['image'], cls, score)

"""# Conversion
For details see
* [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/accessories/ai-camera.html#conversion)
* [Sony IMX500 Converter documentation](https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera/documentation/imx500-converter)
"""

# Convert model
import subprocess
subprocess.run(['imxconv-tf', '-i', MODEL_KERAS, '-o', 'converted', '--overwrite-output'], check=True)

"""
# Expected output from converter:
dnnParams.xml		   mobilenet-quant-rps_MemoryReport.json
mobilenet-quant-rps.pbtxt  packerOut.zip
"""
# List converted files
print("Converted files:")
for file in os.listdir('converted'):
    print(f"  {file}")
assert os.path.exists("converted/packerOut.zip"), f"Converted file not found"

"""# Next step
__OBSERVE__: First, save the quantized model and the output from the conversion to your local machine. For packaging you will need the `packerOut.zip` file.

Next step is to package the model for IMX500, see [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/accessories/ai-camera.html#packaging)
"""