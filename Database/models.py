from sqlalchemy import Column, Integer, String
from .database import Base


class Index(Base):
    __tablename__ = "index"

    id = Column(Integer, primary_key=True, index=True)
    indexId = Column(String, unique=True, index=True)
    indexPath = Column(String, unique=True, index=True)
