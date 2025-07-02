import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

def prepare_image(img, target_size=(128, 128)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)
