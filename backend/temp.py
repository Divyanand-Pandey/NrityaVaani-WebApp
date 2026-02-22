import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("AIzaSyCR10AfIhHvmlRUKK8IVgSneYDkc8ZAAKI"))

for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(model.name)