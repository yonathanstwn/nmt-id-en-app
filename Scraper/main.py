"""
Main runner file to scrape or generate English - Colloquial Indonesian dataset
from OpenAI GPT-3.5 API endpoints.
"""
import openai
import json
from datasets import load_dataset
import sqlalchemy as db
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session


class Base(DeclarativeBase):
    """Base table class used by sqlalchemy to initialise database tables"""
    pass


class Translation(Base):
    """
    Database table schema containing 4 columns:
    - id -> primary key for each translation record/row
    - english -> the English sentence
    - formal_indo -> the formal Indonesian sentence
    - colloquial_indo -> the colloquial Indonesian sentence
    """

    __tablename__ = "translation"

    id: Mapped[int] = mapped_column(primary_key=True)
    english: Mapped[str] = mapped_column(db.String(150))
    formal_indo: Mapped[str] = mapped_column(db.String(150))
    colloquial_indo: Mapped[str] = mapped_column(db.String(150))

    def __repr__(self) -> str:
        return f"Translation(id={self.id!r}, english={self.english!r}, formal_indo={self.formal_indo!r}, colloquial_indo={self.colloquial_indo!r})"


def init_database():
    """
    Create local sqlite database to store the GPT generated translations.
    Session object is then used to interact with the database.
    """
    # Database connection
    engine = db.create_engine("sqlite:///db.sqlite", echo=True)
    # Create database tables
    Base.metadata.create_all(engine)
    # Create session object to interact with database
    session = Session(engine)
    return session


def setup_openai_key():
    """Setup OpenAI API key from api_key.json"""
    with open('api_key.json', 'r') as f:
        openai.api_key = json.load(f)['key']


def load_open_subtitles_dataset(start, end):
    """Load OpenSubtitles dataset with start and end index"""
    return load_dataset("open_subtitles", lang1="en", lang2="id")[
        'train'].select(range(start, end))['translation']


def send_gpt_prompt(english_sentence):
    """Send API request to OpenAI GPT-3.5-turbo model to translate English sentence to Colloquial Indonesian"""
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "English to informal colloquial Indonesian translator. No extra output information."},
            {"role": "user", "content": f"Translate: {english_sentence}"}
        ]
    )


def insert_to_database(session, english_sentence, formal_indo_sentence, colloquial_indo_sentence):
    """Add translation record/row based on parameters to the database"""
    session.add(Translation(english=english_sentence,
                formal_indo=formal_indo_sentence, colloquial_indo=colloquial_indo_sentence))
    session.commit()


def main():
    """Main program to be run when running this file"""

    # Max sentence length so does not incur excessive OpenAI API credits
    CHAR_LIMIT = 100

    session = init_database()
    setup_openai_key()
    dataset = load_open_subtitles_dataset(0, 25_000)
    dataset_size = len(dataset)

    success_count = 0
    skipped_count = 0

    for example in dataset:
        english_sentence = example['en']
        formal_indo_sentence = example['id']
        if len(english_sentence) <= CHAR_LIMIT:
            response = send_gpt_prompt(english_sentence)
            colloquial_indo_sentence = response['choices'][0]['message']['content']
            insert_to_database(session, english_sentence, formal_indo_sentence, colloquial_indo_sentence)
            success_count += 1
        else:
            skipped_count += 1
        
        # Logging progress
        print(f"skipped: {skipped_count}, success: {success_count}, total: {dataset_size}")

        

if __name__ == '__main__':
    main()
