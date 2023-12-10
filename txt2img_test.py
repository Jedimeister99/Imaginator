from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo")
try: 
    pipe.to("cuda")
except:
    pipe.to("cpu")
    
prompt = "village in late summer, river and plain, mountains, pebbled boulders, blue water, troops marching, dusty trees, soldiers marching along road, crops rich with fruit trees, battle in the mountains, artillery flashes, cool nights, highly detailed, dramatic lighting"
results = pipe(
    prompt=prompt,
    num_inference_steps=2,
    guidance_scale=0.0,
)
imga = results.images[0]
imga.save("image.png")