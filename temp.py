import sqlalchemy as db
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy import select

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


def insert_to_database(session, english_sentence, formal_indo_sentence, colloquial_indo_sentence):
    session.add(Translation(english=english_sentence,
                formal_indo=formal_indo_sentence, colloquial_indo=colloquial_indo_sentence))
    session.commit()

engine1 = db.create_engine("sqlite:///Scraper/db.sqlite", echo=True)
session1 = Session(engine1)

engine2 = db.create_engine("sqlite:///Scraper6/db.sqlite", echo=True)
session2 = Session(engine2)

stmt = select(Translation)
for row in session2.scalars(stmt):
    insert_to_database(session1, row.english, row.formal_indo, row.colloquial_indo)

