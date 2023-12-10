# Imaginator
This is a project for combining a large language model with an OCR and Stable Diffusion, to create a image-to-text-to-image pipeline. The project produces a Gradio link that can be hosted on a single computer. In this Gradio UI, you can upload an image of a passage of text, most recommended is a fiction novel. The pipeline will then process the image and convert the text into plaintext, convert this to a summary using a fine-tuned LLM, and pass it to a Stable Diffusion model to create an image based on the passage of text passed in.


# Prerequisites
CUDA-capable NVIDIA GPU is strongly recommended. While this project can support CPU-only mode, it has not been tested and will require manual changes to the code to run normally. The models used for this project are fairly light, so you shouldn't need a beefy card.

# How to install
    - Make sure you have at least Python version 3.10.10 installed and Git.
    - Create a Python virtual environment. ``python -m venv Imagination``
    - Enter your new environment. ``source Imagination/bin/activate``
    - Install the required dependencies. ``pip install -r requirements.txt``
    - Run the project script. ``python imaginator.py``

Enjoy! The script will automatically download needed models and output a link in the terminal for you to visit. 
