import gradio as gr
import torch
from transformers import pipeline
from PIL import Image
import pytesseract
import numpy as np
from clip_interrogator import Interrogator, Config
import timeit
import cv2
import json
import requests
import io
import base64
from diffusers import DiffusionPipeline

pytesseract.pytesseract.tesseract_cmd = ('/usr/bin/tesseract')

def summarize(long_text):
    summarizer = pipeline(
        "summarization",
        "zohfur/longt5-stable-diffusion-prompt",
        device=0 if torch.cuda.is_available() else -1,
    )
    params = {
        "num_beams": 4,
    } #recommended params can be found at https://colab.research.google.com/gist/pszemraj/d9a0495861776168fd5cdcd7731bc4ee/example-long-t5-tglobal-base-16384-book-summary.ipynb
    result = summarizer(long_text, **params)
    return result[0]["summary_text"]

def ocr(photo):
    # apply grayscale, gaussian blur, and otsu's threshold to clean up
    # https://stackoverflow.com/questions/37745519/use-pytesseract-ocr-to-recognize-text-from-an-image
    input_photo = cv2.imread(photo)
    gray = cv2.cvtColor(input_photo, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    #remove noise and invert image
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - thresh
    ocr_output = pytesseract.image_to_string(invert, lang='eng')
    return ocr_output, gray, blur, thresh, invert

def img2txt(input_image):
    conv_image = Image.open(input_image).convert('RGB')
    # reduce image size to speed up inference
    if conv_image.size[0] > 512 or conv_image.size[1] > 512:
        conv_image = conv_image.resize((512, 512), resample=Image.BILINEAR)
    ci = Interrogator(Config(clip_model_name="ViT-L-14/openai")) #ViT-H-14/laion2b_s32b_b79k is better but inference was taking like 45 minutes on a 2070S
    return ci.interrogate(conv_image)


def txt2img(input_prompt):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo")
    try: 
        pipe.to("cuda")
    except:
        pipe.to("cpu")
    image = pipe(
        prompt=input_prompt,
        num_inference_steps=2,
        guidance_scale=0.0,
    )
    image = image.images[0]
    return image
    
def book_to_image(input_summarize_image,):
    ocr_text = ocr(input_summarize_image)[0]
    input_prompt = summarize(ocr_text)
    final_image = txt2img(input_prompt)
    return input_prompt, final_image

# sum_ui = gr.Interface(fn=summarize, inputs=gr.Textbox(lines=6, placeholder="Text to summarize"), outputs="text")
with gr.Blocks() as app:
    with gr.Tab("Summarize"):
        long_text = gr.Textbox(lines=6, placeholder="Text to summarize")
        sum_output = gr.Textbox(lines=6, placeholder="Summary")
        run_summarizer = gr.Button("Summarize")
    with gr.Tab("Img2txt"):
        input_image = gr.Image(type="filepath", label="Image to interrogate")
        img2txt_output = gr.Textbox(lines=6, placeholder="Interrogation")
        img2txt_btn = gr.Button("Interrogate")
    with gr.Tab("OCR"):
        photo = gr.Image(type="filepath", height=250, label="Image to OCR")
        ocr_output = gr.Textbox(lines=8, placeholder="OCR Output")
        ocr_btn = gr.Button("OCR")
        with gr.Row(equal_height=True):
            gray_photo = gr.Image(type="numpy", height=250, label="Grayscale")
            blur_photo = gr.Image(type="numpy", height=250, label="Gaussian Blur")
            thresh_photo = gr.Image(type="numpy", height=250, label="Threshold")
            invert_photo = gr.Image(type="numpy", height=250, label="Invert")
    with gr.Tab("Book to image"):
        page = gr.Image(type="filepath", height=250, label="Image to OCR")
        generated_prompt = gr.Textbox(lines=3, placeholder="Generated Prompt")
        generated_image = gr.Image(type="numpy", height=512, label="Generated Image")
        bti_btn = gr.Button("Generate Artwork")
    img2txt_btn.click(fn=img2txt, inputs=input_image, outputs=img2txt_output, api_name="img2txt")
    run_summarizer.click(fn=summarize, inputs=long_text, outputs=sum_output, api_name="summarize")
    ocr_btn.click(
        fn=ocr,
        inputs=photo,
        outputs=[ocr_output,gray_photo,blur_photo,thresh_photo,invert_photo],
        api_name="ocr")
    bti_btn.click(fn=book_to_image, inputs=[page], outputs=[generated_prompt,generated_image], api_name="bti")
app.launch(share=True)

#ocr -> text -> summarize -> txt2img
