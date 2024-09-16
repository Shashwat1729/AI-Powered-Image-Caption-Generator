# AI-Powered Image Caption Generator

This project generates natural language captions for uploaded images using a combination of **CNNs** for image feature extraction and **LLMs** for text generation. The model used for caption generation is the `vit-gpt2-image-captioning` model from Hugging Face.

## Features:
- Supports image uploads (JPEG, PNG).
- Automatically generates captions for uploaded images using **pre-trained models**.
- A user-friendly web interface built with Flask.

## Setup:

1. Clone this repository:
   ```bash
   git clone https://github.com/Shashwat1729/AI-Powered-Image-Caption-Generator.git
   cd AI-Powered-Image-Caption-Generator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to `http://127.0.0.1:5000/` to use the image caption generator.
