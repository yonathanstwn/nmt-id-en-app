"""
Main runner file to scrape or generate English - Colloquial Indonesian dataset
from OpenAI GPT-3.5 API endpoints.
"""
import openai
import json
from datasets import load_dataset


def main():
    """Main program to be run when running this file"""

    # Load OpenSubtitles dataset
    dataset = load_dataset("open_subtitles", lang1="en", lang2="id")[
        'train'].select(range(50_000))

    # Setup OpenAI API key
    with open('api_key.json', 'r') as f:
        openai.api_key = json.load(f)['key']

    WORD_LIMIT = 20

    # Send prompt as an API request
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "English to colloquial Indonesian translator."},
            {"role": "user", "content": "I want to go to the beach with my friends after I study."}
        ]
    )

    print(response)


if __name__ == '__main__':
    main()
