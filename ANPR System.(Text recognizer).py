import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import pytesseract as tess
tess.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
import os

import re
import cv2
import numpy as np
import string
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from collections import Counter
from PIL import Image
from itertools import groupby

from IPython.display import display
from random import randrange

# import craft functions
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)
img_path = "Crop/image_crops/"
imgs_paths = os.listdir("Crop/image_crops")
custom_config = r'--oem 3 --psm 6'
txt_dict = dict()

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key) 

def detector_recognizer(image_path):
    data_folder ='Crops'
    output_dir = 'Crop/'
    
    # read image
    image = read_image(image_path)
    # load models
    refine_net = load_refinenet_model(cuda=False)
    craft_net = load_craftnet_model(cuda=False)

    # perform prediction
    prediction_result = get_prediction(
      image=image,
      craft_net=craft_net,
      refine_net=refine_net,
      text_threshold=0.5,
      link_threshold=0.2,
      low_text=0.2,
      #cuda=True,
      long_size=1280
    )

    # export detected text regions
    exported_file_paths = export_detected_regions(
       image=image,
       regions=prediction_result["boxes"],
       output_dir=output_dir,
       rectify=True
    )

    empty_cuda_cache()

root = tk.Tk()
def open_file_browser():
    # Open a file browser and ask the user to select a file
    file_path = filedialog.askopenfilename()  # Returns the selected file path
    root.withdraw()
    # If a file was selected, do something with the file_path
    if file_path:
        print(f"You selected: {file_path}")
        # Create a Path object and get the suffix
        ext = Path(file_path).suffix

        # Check if the suffix is one of the common image file extensions
        if ext.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
            print(f"The file {file_path} is an image file.")
            return file_path
        else:
            print(f"The file {file_path} is not an image file.")
    else:
        print("No file was selected.")


image_path = open_file_browser()
root.destroy()


def is_background_white(image, threshold=240, coverage_ratio=0.95):

    # Define the regions (borders) to check for white background
    height, width, _ = image.shape
    border_size = 10  # Define the border width to check

    # Extract border regions
    top_border = image[:border_size, :]
    bottom_border = image[-border_size:, :]
    left_border = image[:, :border_size]
    right_border = image[:, -border_size:]

    # Combine all border regions into one image
    border = np.vstack((top_border, bottom_border, left_border, right_border))

    # Convert to grayscale
    gray_border = cv2.cvtColor(border, cv2.COLOR_BGR2GRAY)

    # Create a binary mask where "white" regions are marked
    _, binary_mask = cv2.threshold(gray_border, threshold, 255, cv2.THRESH_BINARY)

    # Calculate the ratio of white pixels to the total number of pixels in the borders
    white_pixel_ratio = np.sum(binary_mask == 255) / binary_mask.size

    # Determine if the background is considered white based on the coverage ratio
    return white_pixel_ratio >= coverage_ratio

def convert_background_to_white(image):

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to isolate non-background regions
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the non-background regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask where the background is white and non-background regions are black
    mask = np.ones_like(image) * 255  # start with a white image
    cv2.drawContours(mask, contours, -1, (0, 0, 0), thickness=cv2.FILLED)

    # Combine the original image and the mask
    background_white = np.where(mask == 0, image, (255, 255, 255))
    return background_white


detector_recognizer(image_path)

for path in imgs_paths:
    img = cv2.imread(img_path + path)
    if is_background_white(img) == False:
        img = convert_background_to_white(img)
    x = img.shape[0]
    y = img.shape[1]
    text = tess.image_to_string(img, config=custom_config, output_type=tess.Output.DICT,lang='eng')
    img1 = cv2.imread(image_path)
    txt = text.get('text')
    txt_dict[txt] = len(txt)
max_key = max(txt_dict, key=txt_dict.get)
# Window name in which image is displayed
window_title = 'Image'

# font
f = cv2.FONT_HERSHEY_SIMPLEX

# point
p = (x//2, y//2)

# font Scale
fScale = 1

# Blue color in BGR
c = (255, 0, 0)

# Line thickness of 2 px
t = 2

# Using cv2.putText() method
image = cv2.putText(img1, max_key, (p), f, 
                   fScale, c, t, cv2.LINE_AA)

# Displaying the image
cv2.imshow(window_title, img1) 
cv2.waitKey(0)
cv2.destroyWindow("Image")
print("Number  Detected Plate Text : ",max_key)

