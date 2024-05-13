import os
import requests
import gradio as gr
from transformers import pipeline
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate, LLMChain

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name="gpt-3.5-turbo", temperature=0.3)

# Image to Text
def image_to_text(url):
    # Load a transformer
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]['generated_text']
    
    return text

# Generate Story
def generate_story(scenario):
    template = """
    you are a very good story teller and a very nice person:
    you can generate a short fairy tail based on a single narrative, the story should take 60 seconds to read.
    CONTEXT: {scenario}
    STORY:
    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    story_llm = LLMChain(llm=llm, prompt=prompt, verbose=True)
    story = story_llm.predict(scenario=scenario)
    return story

# Text to Speech
def text_to_speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_TOKEN')}"}
    payload = {"inputs": message}

    response = requests.post(API_URL, headers=headers, json=payload)
    with open('audio.mp3', 'wb') as audio_file:
        audio_file.write(response.content)

def image_storyteller(image):
    scenario = image_to_text(image)
    story = generate_story(scenario)
    text_to_speech(story)
    return scenario, story, "audio.mp3"

inputs = gr.Image(type='pil', label="Upload an Image")
outputs = [gr.Textbox(label="Scenario"), gr.Textbox(label="Story"), gr.Audio(label="Audio")]

title = "Image Storyteller"
description = "Upload an image and generate a story based on the image content."
examples = ["img.jpg"]

gr.Interface(fn=image_storyteller, inputs=inputs, outputs=outputs, title=title, description=description, examples=examples).launch()
