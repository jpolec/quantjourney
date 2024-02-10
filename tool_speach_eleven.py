"""
    Tool to generate speech using the Eleven API
    
    author: jpolec@gmail
    date: 2024-01-21
"""

import os
import json
import http.client

# Function to prompt user for API key and save it in a file
def get_api_key():
    api_key = input("Enter your API key: ")
    with open("api_key.txt", "w") as file:
        file.write(api_key)

# Function to load the API key from the file
def load_api_key():
    try:
        with open("api_key.txt", "r") as file:
            api_key = file.read().strip()
        return api_key
    except FileNotFoundError:
        return None

# Function to prompt user for input
def get_text():
    return input("Enter the text you want to convert to speech: ")

# Function to prompt user for output file name
def get_output_filename():
    return input("Enter the desired output file name (without extension): ")


# Function to prompt user to select a voice
def select_voice(voices):
    print("Available voices:")
    for index, voice in enumerate(voices):
        print(f"{index+1}. {voice['name']}")
    voice_index = int(input("Enter the number corresponding to the desired voice: ")) - 1
    return voices[voice_index]['voice_id']

# Function to send a request to the API and save the speech
def generate_speech(api_key, text, voice_id, output_directory, output_filename):
    # API endpoint
    conn = http.client.HTTPSConnection("api.elevenlabs.io")

    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    # Request headers
    headers = {
        "accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }

    # Request payload
    payload = {
        "text" : text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0
        }
    }

    # Send POST request to the API
    conn.request("POST", f"/v1/text-to-speech/{voice_id}?optimize_streaming_latency=0", headers=headers, body=json.dumps(payload))

    # Get the response
    response = conn.getresponse()

    # Check if the request was successful
    if response.status == 200:
        # Save the audio file
        with open(os.path.join(output_directory, f"{output_filename}.mp3"), "wb") as file:
            file.write(response.read())
        print("Speech generated successfully!")
    else:
        print("Error:", response.status)

    # Close the connection
    conn.close()

def text_to_speech(api_key=None, text="", voice_id="", output_directory=".", output_filename="output"):
    """
    """
    
    # Load API key from file if not provided
    if api_key is None:
        api_key = load_api_key_from_file()
        if api_key is None:
            return "API key is required."

    # Define the available voices
    voices = [
        {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel"},
        {"voice_id": "AZnzlk1XvdvUeBnXmlld", "name": "Domi"},
        {"voice_id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella"},
        {"voice_id": "ErXwobaYiN019PkySvjV", "name": "Antoni"},
        {"voice_id": "MF3mGyEYCl7XYWbV9V6O", "name": "Elli"},
        {"voice_id": "TxGEqnHWrfWFTfGW9XjX", "name": "Josh"},
        {"voice_id": "VR6AewLTigWG4xSOukaG", "name": "Arnold"},
        {"voice_id": "pNInz6obpgDQGcFmaJgB", "name": "Adam"},
        {"voice_id": "yoZ06aMxZJJ28mfd3POQ", "name": "Sam"}
    ]

    # Check if the specified voice_id is valid
    if voice_id not in [voice['voice_id'] for voice in voices]:
        return "Invalid voice ID. Please select from the available voices."

    # Generate the speech using the API key, text, and voice ID
    generate_speech(api_key, text, voice_id, output_directory, output_filename)


def main():
    
    if api_key is None:
    # Prompt user for API key and save it
        get_api_key()
        print("API key saved successfully!")
        # Load the API key from the file
        api_key = load_api_key()
    else:
        print("API key loaded successfully!")

    # Prompt user for input
    text = get_text()

    # Define the available voices
    voices = [
        {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel"},
        {"voice_id": "AZnzlk1XvdvUeBnXmlld", "name": "Domi"},
        {"voice_id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella"},
        {"voice_id": "ErXwobaYiN019PkySvjV", "name": "Antoni"},
        {"voice_id": "MF3mGyEYCl7XYWbV9V6O", "name": "Elli"},
        {"voice_id": "TxGEqnHWrfWFTfGW9XjX", "name": "Josh"},
        {"voice_id": "VR6AewLTigWG4xSOukaG", "name": "Arnold"},
        {"voice_id": "pNInz6obpgDQGcFmaJgB", "name": "Adam"},
        {"voice_id": "yoZ06aMxZJJ28mfd3POQ", "name": "Sam"}
    ]
    # Prompt user to select a voice
    voice_id = select_voice(voices)

    # Prompt user for output file name
    output_filename = get_output_filename()

    output_directory = '_voice'
    
    # Generate the speech using the API key, text, and voice ID
    generate_speech(api_key, text, voice_id, output_directory, output_filename)
    
    exit()

if __name__ == "__main__":
    
    api_key = ""


    # Example of using the function
    text_to_speech(
        api_key=api_key, 
        text=""
        voice_id=""
        output_directory="_voice",  # Specify your output directory
        output_filename="my_speech"  # Specify your desired output filename without extension
    )
    
    #main()
    
