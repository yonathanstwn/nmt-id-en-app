"""
Main runner file to scrape or generate English - Colloquial Indonesian dataset
from OpenAI GPT-3.5 API endpoints.
"""
from http.client import RemoteDisconnected
import openai
import json
from datasets import load_dataset
import sqlalchemy as db
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy import select
import signal


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


def print_database():
    """Utility function to view database during execution"""
    # Database connection
    engine = db.create_engine("sqlite:///db.sqlite", echo=True)
    # Create session object to interact with database
    session = Session(engine)
    # View database
    stmt = select(Translation)
    for row in session.scalars(stmt):
        print(row)


def timeout(n_seconds):
    """Higher level function or decorator. Gives corresponding function call a timeout."""
    
    def process(f):
        """Inner function to process the actual timeout functionality"""

        def handle(signum, frame):
            """Executes when timeout is triggered, i.e. when function executes for more than {n_seconds} seconds"""
            raise TimeoutError(f"TIMEOUT: function has been executing more than {n_seconds} sec.")
        
        def f_wrapper(*args, **kwargs):
            """Wrapper for the actual function f to be given a timeout"""
            signal.signal(signal.SIGALRM, handle)
            signal.alarm(n_seconds)
            try:
                return f(*args, **kwargs)
            finally:
                signal.alarm(0)
        
        return f_wrapper
    
    return process


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


@timeout(n_seconds=3)
def send_gpt_prompt(english_sentence):
    """
    Send API request to OpenAI GPT-3.5-turbo model to translate English sentence to Colloquial Indonesian.
    Implements a safety try-except block to handle bad connection issues which simply returns None if not successful.
    """
    try:
        x = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "English to informal colloquial Indonesian translator. No extra output information."},
                {"role": "user", "content": f"Translate: {english_sentence}"}
            ]
        )
        print(x)
        return x
    except Exception as e:
        print(f"FAILED with exception: {e}\nTrying again...")
        return None

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
    dataset = load_open_subtitles_dataset(2_200_000, 2_300_000)
    dataset_size = len(dataset)

    success_count = 0
    skipped_count = 0

    for example in dataset:
        english_sentence = example['en']
        formal_indo_sentence = example['id']
        if len(english_sentence) <= CHAR_LIMIT:
            response = None
            # Keep trying to send request until successful
            while response is None:
                print("\nTrying to send request...")
                try:
                    response = send_gpt_prompt(english_sentence)
                except TimeoutError as e:
                    print(f"{e}\nTrying again...")
                    response = None
            colloquial_indo_sentence = response['choices'][0]['message']['content']
            insert_to_database(session, english_sentence,
                               formal_indo_sentence, colloquial_indo_sentence)
            success_count += 1
        else:
            skipped_count += 1

        # Logging progress
        print(
            f"skipped: {skipped_count}, success: {success_count}, total: {dataset_size}")


if __name__ == '__main__':
    main()
