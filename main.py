import streamlit as st
from PIL import Image
from transformers import pipeline
import os
from paddleocr import PaddleOCR, draw_ocr
import time

class ImageOCR:
    def __init__(self):
        """
        Initialize the object with an instance of PaddleOCR with GPU disabled and angle classification disabled.
        """
        self.ocr = PaddleOCR()

    def extract_text(self, image_path):
        """
        Extracts text from the image located at the specified image_path using OCR.
        :param image_path: str - the path to the image file to extract text from
        :return: str - the extracted text from the image, or None if an exception occurs
        """
        try:
            result = self.ocr.ocr(image_path)
            extracted_text = ' '.join([word[1][0] for line in result for word in line])
            os.remove(image_path)
            return extracted_text
        except Exception as e:
            print(e)
            return None

def main():
    st.sidebar.title('About')
    st.sidebar.write("This is a simple Streamlit app that uses Hugging Face's Transformers library to summarize long articles. The app provides an input section for users to enter their articles, and a 'Summarize' button that generates a summary of the article. The summary is generated using the 'Falconsai/text_summarization' model.")
    st.sidebar.write("Project GitHub: https://github.com/achuajays/AI-PNG-Article-Summarizer")
    st.sidebar.write("Author: Adarsh Ajay")
    
    st.title('Article Summarizer')

    st.write("Upload a PNG image of a long article and click the 'Summarize' button to generate a summary.")
    uploaded_file = st.file_uploader("Upload PNG image", type="png")

    if uploaded_file is not None:
        with st.spinner('Loading...'):
            time.sleep(2)

    ocr = ImageOCR()

    if uploaded_file is not None and st.button("Summarize"):
        with st.spinner('Summarizing...'):
            image = Image.open(uploaded_file)
            image.save('article.png')

            extracted_text = ocr.extract_text('article.png')

            if extracted_text:
                summarizer = pipeline("summarization", model="Falconsai/text_summarization")
                summary = summarizer(extracted_text, max_length=100, min_length=30, do_sample=False)
                st.success('Summary:')
                st.write(summary[0]['summary_text'])

    

if __name__ == '__main__':
    main()


