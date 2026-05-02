import base64
import json
from datetime import datetime
import gradio as gr
import torch
import spaces
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import ast
import os
from datetime import datetime
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files


# Define constants
DESCRIPTION = "[ShowUI Demo](https://huggingface.co/showlab/ShowUI-2B)"
_SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1344 * 28 * 28

# Specify the model repository and destination folder
model_repo = "showlab/ShowUI-2B"
destination_folder = "./showui-2b"

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# List all files in the repository
files = list_repo_files(repo_id=model_repo)

# Download each file to the destination folder
for file in files:
    file_path = hf_hub_download(repo_id=model_repo, filename=file, local_dir=destination_folder)
    print(f"Downloaded {file} to {file_path}")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "./showui-2b",
    # "showlab/ShowUI-2B",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

# Load the processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

def draw_point(image_input, point=None, radius=5):
    """Draw a point on the image."""
    if isinstance(image_input, str):
        image = Image.open(image_input)
    else:
        image = Image.fromarray(np.uint8(image_input))

    if point:
        x, y = point[0] * image.width, point[1] * image.height
        ImageDraw.Draw(image).ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')
    return image

def array_to_image_path(image_array):
    """Save the uploaded image and return its path."""
    if image_array is None:
        raise ValueError("No image provided. Please upload an image before submitting.")
    img = Image.fromarray(np.uint8(image_array))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.png"
    img.save(filename)
    return os.path.abspath(filename)

@spaces.GPU
def run_showui(image, query):
    """Main function for inference."""
    image_path = array_to_image_path(image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _SYSTEM},
                {"type": "image", "image": image_path, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
                {"type": "text", "text": query}
            ],
        }
    ]

    # Prepare inputs for the model

    global model

    model = model.to("cuda")

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda")

    # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Parse the output into coordinates
    click_xy = ast.literal_eval(output_text)

    # Draw the point on the image
    result_image = draw_point(image_path, click_xy, radius=10)
    return result_image, str(click_xy)


image_path = array_to_image_path(image)
result_image, coordinates = run_showui(image, query)
print("Coordinates:", coordinates)
print("Image Path:", image_path)