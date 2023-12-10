import requests
import csv
import os
from datasets import load_dataset
import time
import openai
from openai import OpenAI
#from tenacity import (retry,stop_after_attempt,wait_random_exponential)

client = OpenAI(api_key='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')
# Load the booksum dataset
dataset = load_dataset('kmfoda/booksum')

# File to keep track of processed chapters
processed_chapters_file = 'processed_chapters.txt'

system = """
You are a text-to-image prompt generator. Each generation is called a summarization. 
Generate a summarization of the input text using 'tags', following the formula below. 
A summarization is a list of tags, separated by commas. 
Each summarization should reasonably include at least 100 tokens of text. 
Each tag is a short string of words separated by commas.

Each summarization should additionally include words from this list: masterpiece, digital painting, dramatic lighting, highly detailed, 8k uhd, global illumination
IMPORTANT, Make sure to visually describe any people, objects, facial expressions, poses, and scenery in vivid detail. Such as, "golfer with plaid shirt and cap holding iron", "little girl in red clock holding wicker basket of fruit".
Here are a few examples of summarizations:
[the street of a medieval fantasy town, at dawn, dark, 4k, highly detailed, masterpiece, realistic lighting, paved road, medieval buildings, masterpiece, digital painting, dramatic lighting, 8k uhd, highly detailed]
[a highly detailed matte painting of a man on a hill watching a rocket launch in the distance by studio ghibli, makoto shinkai, by artgerm, by wlop, by greg rutkowski, volumetric lighting, octane render, 4 k resolution, trending on artstation, masterpiece, hyperrealism, highly detailed, insanely detailed, intricate, cinematic lighting, depth of field]
 
You will respond with the summarization and no other text or response. Do not print an acknowledgement of the prompt or any extra formatting.
"""

# Load processed chapters into a set for quick lookup
# processed_chapters = set()
# if os.path.exists(processed_chapters_file):
#     with open(processed_chapters_file, 'r') as f:
#         for line in f:
#             processed_chapters.add(line.strip())

# Open the CSV file
with open('training_dataset.csv', 'a', newline='') as csvfile:
    # Create a CSV writer
    writer = csv.writer(csvfile)
    
    # Write the header row if the file is empty
    if os.path.getsize('training_dataset.csv') == 0:
        writer.writerow(['input', 'result'])
    
    # Iterate through the dataset
    for book in dataset['train']:
        # Grab a chapter string
        chapter = book['chapter']
        chapter = chapter[3000:8000]
        
        # If the chapter has already been processed, skip it
        # if book['book_id'] in processed_chapters:
        #     continue
        
        
        print("Sending text:" + chapter[0:60] + "...")

        prompt = "[INPUT TEXT: \n]" + str(chapter)
        try:
            # Call the web API
            # response = requests.get(f'http://100.109.178.56:3000/ask?prompt=[{{"role":"user","content":"{prompt}"}}]&site=you')
            
            # Call the ChatGPT API
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract the result string from chatgpt
            result_string = completion.choices[0].message.content
            result_string = result_string.strip('"')
            result_string = result_string.strip('[')
            result_string = result_string.strip(']')
            print(f"Got response: {result_string}")
            # Raise an exception if the request was unsuccessful from webapi
            #response.raise_for_status()

            # Extract the result string from webapi
            #result_string = response.json()['content']

            # Append to the CSV file
            writer.writerow([chapter, result_string])
            
            # Add the chapter to the processed chapters and write it to the file
            # processed_chapters.add(book['book_id'])
            # with open(processed_chapters_file, 'a') as f:
            #    f.write(book['book_id'] + '\n')
        
        except openai.APIError as e:
            #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            pass
        except openai.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
            pass
        except openai.RateLimitError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass
        #time.sleep(1)
print("Completed!")