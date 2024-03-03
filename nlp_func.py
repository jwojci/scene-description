import os
import streamlit as st
from openai import OpenAI


def get_gpt_client():
    client = OpenAI(api_key=st.secrets["open_ai_key"])
    return client


def get_background_foreground(object_locations):
    background_detections = {}
    objects = {}
    for key in object_locations:
        if not object_locations[key]["isthing"]:
            background_detections[key] = object_locations[key]
        else:
            objects[key] = object_locations[key]
    return background_detections, objects


def get_gpt_prompt(background_detections, objects):
    background_content = f"In the background you'll find: {background_detections}"
    foreground_content = f"Detected objects: {objects}"
    return background_content, foreground_content


def get_description(background_content, foreground_content):
    client = get_gpt_client()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "Given panoptic segmentation samples in a JSON format for an image, generate a concise and informative description of the scene. The JSON file will be provided in two parts. One contains background information and the other contains the detected objects and their bounding boxes every 15 frames. Ensure the output highlights significant elements like people, objects, and surroundings, providing a clear description of the visual content. Try to interpret the bounding boxes and how they change over time to estimate the motion, location etc. (But don't mention the bounding boxes). Don't make things up."},
            {"role": "user", "content": background_content},
            {"role": "user", "content": foreground_content}
        ],
        stream=True
    )

    return response
