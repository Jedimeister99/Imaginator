# Imaginator v2.0
This is a project for combining a large language model with an OCR and Stable Diffusion, to create a image-to-text-to-image pipeline. The project produces a Gradio link that can be hosted on a single computer. In this Gradio UI, you can upload an image of a passage of text, most recommended is a fiction novel. The pipeline will then process the image and convert the text into plaintext, convert this to a summary using a fine-tuned LLM, and pass it to a Stable Diffusion model to create an image based on the passage of text passed in.

#Model & Dataset
The model used is version 2 of this: https://huggingface.co/vahn9995/longt5-stable-diffusion-prompt

Which was trained on the dataset created by this project: https://huggingface.co/datasets/vahn9995/booksum-stable-diffusion-prompt


# Prerequisites
CUDA-capable NVIDIA GPU is strongly recommended. While this project can support CPU-only mode, it has not been tested and will require manual changes to the code to run normally. The models used for this project are fairly light, so you shouldn't need a beefy card.

# How to install

Install required dependencies. <br />
sudo apt install wget lzma liblzma-dev build-essential libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev nvidia-cuda-toolkit<br />
    - Build Python 3.10 from source.<br />
    - wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz<br />
    - tar xzf Python-3.10.13.tgz<br />
    - cd Python-3.10.13<br />
    - ./configure --enable-optimizations<br />
    - sudo make altinstall <br />

Update pip. using: pip3.10 install --upgrade pip

    - Create a Python virtual environment. python3.10 -m venv Imagination
    - Enter your new environment. source Imagination/bin/activate
    - Install the required dependencies. pip3.10 install -r requirements.txt
    - Run the project script. python imaginator.py
    - Enjoy! The script will automatically download needed models and output a link in the terminal for you to visit. 
