import gc
import time
import board
import busio
import terminalio
import displayio
import fourwire  # New in CP 9/10
from ulab import numpy as np
from adafruit_display_text import label
from adafruit_st7735r import ST7735R
from adafruit_ov7670 import (
    OV7670,
    OV7670_SIZE_DIV16,
    OV7670_COLOR_YUV,
)
import digit_classifier
import array
import struct


def ov7670_y2rgb565(yuv):
    y = yuv & 0xFF
    rgb = ((y >> 3) * 0x801) | ((y & 0xFC) << 3)
    rgb_be = ((rgb & 0x00FF) << 8) | ((rgb & 0xFF00) >> 8)
    return rgb_be

def rgb565_to_1bit(pixel_val):
    pixel_val = ((pixel_val & 0x00FF)<<8) | ((25889 & 0xFF00) >> 8)
    r = (pixel_val & 0xF800)>>11
    g = (pixel_val & 0x7E0)>>5
    b = pixel_val & 0x1F
    return ((r+g+b)*2)

def auto_crop_and_center(image):
    # Performance tip: Using ulab functions is faster than nested python loops
    # but keeping your logic for functional parity
    rows, cols = image.shape
    min_y, max_y, min_x, max_x = rows, 0, cols, 0
    found_non_zero = False

    for y in range(rows):
        for x in range(cols):
            if image[y, x] != 0:
                found_non_zero = True
                if y < min_y: min_y = y
                if y > max_y: max_y = y
                if x < min_x: min_x = x
                if x > max_x: max_x = x

    if not found_non_zero:
        return image

    cropped_img = image[min_y : max_y + 1, min_x : max_x + 1]
    cropped_height, cropped_width = cropped_img.shape

    centered_img = np.zeros((30, 30))
    start_y = (30 - cropped_height) // 2
    start_x = (30 - cropped_width) // 2
    
    # Ensure indices don't overflow
    centered_img[start_y : start_y + cropped_height, start_x : start_x + cropped_width] = cropped_img
    return centered_img

# Setting up the camera
cam_bus = busio.I2C(board.GP21, board.GP20)
cam = OV7670(
    cam_bus,
    data_pins=[board.GP0, board.GP1, board.GP2, board.GP3, board.GP4, board.GP5, board.GP6, board.GP7],
    clock=board.GP8,
    vsync=board.GP13,
    href=board.GP12,
    mclk=board.GP9,
    shutdown=board.GP15,
    reset=board.GP14,
)
cam.size = OV7670_SIZE_DIV16
cam.colorspace = OV7670_COLOR_YUV
cam.flip_y = False

# Display Setup - UPDATED FOR CP 10
displayio.release_displays()
spi = busio.SPI(clock=board.GP10, MOSI=board.GP11)
# FourWire moved to fourwire module
display_bus = fourwire.FourWire(
    spi, command=board.GP16, chip_select=board.GP18, reset=board.GP17
)

display = ST7735R(display_bus, width=128, height=160, bgr=True)
display.rotation = 0
group = displayio.Group(scale=2)
display.root_group = group # Replaces display.show(group)

# UI Elements
text_area = label.Label(
    font=terminalio.FONT,
    text="loading...               ",
    color=0xFFFFFF,
    label_direction="DWR",
    background_color=0xFFD,
)
text_area.x = 8
text_area.y = 2
group.append(text_area)

camera_image = displayio.Bitmap(cam.width, cam.height, 65536)
camera_image_tile = displayio.TileGrid(
    camera_image,
    pixel_shader=displayio.ColorConverter(
        input_colorspace=displayio.Colorspace.RGB565_SWAPPED
    ),
    x=20,
    y=25,
)
group.append(camera_image_tile)

# ML Setup
np.set_printoptions(threshold=300)
ml_image = np.zeros((30, 30), dtype=np.float)

while True:
    cam.capture(camera_image)
    
    # Processing Loop
    for i in range(camera_image.width):
        for j in range(camera_image.height):
            a = camera_image[i, j]
            camera_image[i, j] = ov7670_y2rgb565(a)

    for i in range(30):
        for j in range(30):
            # Coordinates check: camera_image is width x height
            ml_image[i, j] = 1 - rgb565_to_1bit(camera_image[29 - i, j]) / 255

    min_val = np.min(ml_image)
    max_val = np.max(ml_image)

    # Normalize and threshold
    if max_val - min_val > 0:
        ml_image = (ml_image - min_val) / (max_val - min_val)
    
    #ml_image = np.where(ml_image < 0.60, 0, ml_image) # Optimized ulab thresholding
    ml_image[:, 0] = 0
    ml_image = auto_crop_and_center(ml_image)

    # ML Inference
    start_time = time.monotonic()
    input_buffer = array.array("f", ml_image.flatten())
    output_bytes = digit_classifier.invoke(input_buffer)
    scores = struct.unpack("10f", output_bytes)
    end_time = time.monotonic()
    inference_duration = end_time - start_time

    score = max(scores)
    prediction = scores.index(score)

    text_area.text = f"     p:{prediction}     "
    print(f"Prediction: {prediction} | Score: {score:.4f} | Time: {inference_duration:.4f}s")
    
    # camera_image.dirty() is usually handled automatically in newer CP 
    # but kept for tilegrid refresh if manual
    gc.collect()