"""
Main runner file to scrape or generate English - Colloquial Indonesian dataset
from OpenAI GPT-3.5 API endpoints.
"""
import openai

def main():
    openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"}
        ]
    )


if __name__ == '__main__':
    main()
