# This is a demo of the IMX500 AI camera.

import time
from typing import List, Tuple, Optional

import cv2
import numpy as np

from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
from picamera2.devices.imx500.postprocess import softmax

# Constants
THRESHOLD = 0.9  # Minimum confidence level for drawing the classification result
TEXT_COLOR = (0, 255, 255)  # Yellow color in BGR format
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1
TEXT_THICKNESS = 2
TEXT_PADDING = 5
BACKGROUND_DARKEN_FACTOR = 2  # Factor to darken the background

# Global state
last_detection: Optional[Tuple[int, float]] = None  # store the last detection result
imx500: Optional[IMX500] = None  # IMX500 helper object
intrinsics: Optional[NetworkIntrinsics] = None  # Network configuration object

def parse_and_draw_classification_result(request: CompletedRequest) -> None:
    """
    Parse and draw the classification results from the IMX500 AI unit.

    Args:
        request (CompletedRequest): The completed camera request containing
            metadata and frame data from the camera.
    """
    result = parse_classification_result(request)
    draw_classification_result(request, result)

def parse_classification_result(request: CompletedRequest) -> Tuple[int, float]:
    """
    Extract the classification results from the IMX500 output tensor.

    Args:
        request (CompletedRequest): The completed camera request containing
            metadata and frame data from the camera.

    Returns:
        Tuple[int, float]: A tuple containing:
            - int: The index of the highest confidence class
            - float: The confidence score for the detected class
        Returns the last detection if there is no output tensor for this frame.
    """
    global last_detection
    np_outputs = imx500.get_outputs(request.get_metadata())
    if np_outputs is None:
        # Occasionally, the IMX500 may not have finished processing this frame, in which
        # case we should re-use the last result.
        return last_detection

    np_output = np_outputs[0]
    if intrinsics.softmax:  # this network recommends applying softmax
        np_output = softmax(np_output)

    top_index = np.argmax(np_output)  # get the index of the highest confidence class
    last_detection = (top_index, np_output[top_index])
    return last_detection

def draw_classification_result(
    request: CompletedRequest,
    result: Tuple[int, float]
) -> None:
    """
    Draw classification results on the output frame with a semi-transparent background.

    Args:
        request (CompletedRequest): The completed camera request containing
            the frame data to draw on.
        result (Tuple[int, float]): The classification result containing:
            - int: The index of the detected class
            - float: The confidence score

    Note:
        The function only draws results if the confidence score is above
        the global THRESHOLD value.
    """
    with MappedArray(request, "main") as m:
        # Create and validate the text label
        label = intrinsics.labels[result[0]] if result and result[1] > THRESHOLD else None
        if label is None:
            return
        text = f"{label}: {result[1]:.3f}"

        # Calculate text size and position
        (text_width, text_height), baseline = cv2.getTextSize(
            text, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS
        )
        text_x = TEXT_PADDING
        text_y = text_height + TEXT_PADDING

        # Darken the background where the text will be to make it more readable.
        rectangle = m.array[text_y - text_height:text_y + baseline, text_x:text_x + text_width]
        rectangle[...] = rectangle // BACKGROUND_DARKEN_FACTOR

        # Draw text on top of the background.
        cv2.putText(m.array, text, (text_x, text_y),
                    TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

if __name__ == "__main__":
    # Create the IMX500 helper object first so that it gives us the camera number of the IMX500.
    imx500 = IMX500("network.rpk")

    # The intrinsics contains useful information about the network.
    intrinsics = NetworkIntrinsics()
    intrinsics.task = "classification"
    with open("labels.txt", "r") as f:
        intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    # The intrinsics also contains the inference rate of this network, which should be a good choice
    # for the framerate of the camera.
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate})

    imx500.show_network_fw_progress_bar()  # Show the progress of the network being loaded
    picam2.start(config, show_preview=True)

    # Set the callback function to handle the results of the IMX500 on every frame.
    picam2.pre_callback = parse_and_draw_classification_result

    # Sleep forever while the camera runs.
    while True:
        time.sleep(1)
