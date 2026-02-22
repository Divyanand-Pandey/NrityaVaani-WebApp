import os
import google.generativeai as genai

try:
    with open("api.bin", "r") as f:
        api_key = f.read().strip()
    
    print(f"API Key loaded. Length: {len(api_key)}")
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Say 'Test successful'")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
