import os
from dotenv import load_dotenv

import google.generativeai as genai

import PIL.Image
import pickle

import textwrap
from rich.console import Console
from rich.markdown import Markdown


class PrintMarkdown:
    def __init__(self):
        self.console = Console()

    def _process_text(self, text):
        text = text.replace('•', '  *')
        text = Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
        return text

    def print(self, text):
        text = self._process_text(text)
        self.console.print(text)


class ImageProcessing:
    def __init__(self):
        pass

    def load(self, path):
        return PIL.Image.open(path)


class GenrateGemini():
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.text_model = genai.GenerativeModel('gemini-pro')
        self.vision_model = genai.GenerativeModel('gemini-pro-vision')

        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            }
        ]

    def genrate_text(self, prompt, image=False):
        if image:
            response = self.text_model.generate_content(
                [prompt, image], safety_settings=self.safety_settings)
        else:
            response = self.text_model.generate_content(
                prompt, safety_settings=self.safety_settings)
        return response

    def start_chat_session(self, history,):
        session = self.text_model.start_chat(history=history)
        return session

    def chat(self, prompt, session):
        response = session.send_message(
            prompt, safety_settings=self.safety_settings)
        self.save_history(session=session)
        return response

    def save_history(self, session):
        try:
            os.mkdir("history")
        except Exception as e:
            pass

        with open(f"history/{session.history[0].parts[0].text}.pkl", "wb") as f:
            pickle.dump(session.history, file=f)

    def load_history(self, path):
        with open(path) as f:
            history = pickle.load(f)
        return history


if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("API_KEY")

    markdown = PrintMarkdown()
    gemini = GenrateGemini(api_key=API_KEY)

    session = gemini.start_chat_session([])
    response = gemini.chat("give me the lesson from atomic habbits", session)

    markdown.print(response.text)

    response2 = gemini.chat(
        "give me the lesson from subtle art of not giving a f*ck", session)

    # response = gemini.genrate_text("Give me 10 learnings from bhagwad gita")

    markdown.print(response2.text)
