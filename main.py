import os
from dotenv import load_dotenv

import google.generativeai as genai

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


class GenrateGemini():
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def genrate_text(self, text):
        response = self.model.generate_content(text)
        return response


if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("API_KEY")

    markdown = PrintMarkdown()
    gemini = GenrateGemini(api_key=API_KEY)

    response = gemini.genrate_text("Give me 10 learnings from bhagwad gita")
    markdown.print(response.text)
