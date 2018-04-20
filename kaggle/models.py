from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime

Base = declarative_base()


class LogMessage(Base):
    __tablename__ = 'log'

    id = Column(Integer, primary_key=True, autoincrement=True)
    task = Column(String)
    pid = Column(String)
    iteration = Column(Integer)
    eta = Column(Float)
    norm_grad = Column(Float)
    norm_beta = Column(Float)
    objective = Column(Float)
    timestamp = Column(DateTime)