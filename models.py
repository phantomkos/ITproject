from sqlalchemy import Column, Integer, LargeBinary, String
from database import Base

class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, index=True)
    image_data = Column(LargeBinary)
    category = Column(String)