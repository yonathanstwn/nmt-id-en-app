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
    pass


class Translation(Base):
    __tablename__ = "translation"

    id: Mapped[int] = mapped_column(primary_key=True)
    english: Mapped[str] = mapped_column(db.String(30))
    formal_indo: Mapped[str] = mapped_column(db.String(30))
    colloquial_indo: Mapped[str] = mapped_column(db.String(30))

    def __repr__(self) -> str:
        return f"Translation(id={self.id!r}, english={self.english!r}, formal_indo={self.formal_indo!r}, colloquial_indo={self.colloquial_indo!r})"


def main():
    """Main program to be run when running this file"""

    # Load OpenSubtitles dataset
    dataset = load_dataset("open_subtitles", lang1="en", lang2="id")[
        'train'].select(range(50_000))

    # Create local sqlite database to store the GPT generated translations
    engine = db.create_engine("sqlite:///db.sqlite", echo=True)
    Base.metadata.create_all(engine)
    session = Session(engine)

    # Setup OpenAI API key
    with open('api_key.json', 'r') as f:
        openai.api_key = json.load(f)['key']

    WORD_LIMIT = 20

    # # Send prompt as an API request
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "English to colloquial Indonesian translator."},
    #         {"role": "user", "content": "I want to go to the beach with my friends after I study."}
    #     ]
    # )

    # print(response)

    # Add translation to database
    session.add(Translation(english="yes",
                formal_indo="iya", colloquial_indo="ya"))
    session.commit()


if __name__ == '__main__':
    main()
