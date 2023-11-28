from typing import List

from fastapi import FastAPI, Request, UploadFile, File, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, LargeBinary, String
from sqlalchemy.ext.declarative import declarative_base
from database import SessionLocal
from fastapi.responses import StreamingResponse
import io
import models

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 데이터베이스 모델 정의
Base = declarative_base()

class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True, index=True)
    image_data = Column(LargeBinary)
    category = Column(String)

# 테이블 생성
Base.metadata.create_all(bind=create_engine("sqlite:///images.sqlite3", connect_args={'check_same_thread': False}))
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/uploadfile")
async def create_upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 파일 데이터 읽기
    contents = await file.read()

    filename = file.filename
    category = filename.split(",")[0]

    # 데이터베이스에 저장
    db_image = Image(image_data=contents, category=category)
    db.add(db_image)
    db.commit()
    db.refresh(db_image)

    return {"filename": file.filename, "content_type": file.content_type, "category": category, "id": db_image.id}


@app.get("/")
async def read_root(request: Request, db: Session = Depends(get_db)):
    # 데이터베이스에서 이미지 목록 가져오기
    images = db.query(Image).all()
    return templates.TemplateResponse("index.html", {"request": request, "images": images})

@app.get("/image/{image_id}")
async def read_image(image_id: int, db: Session = Depends(get_db)):
    image = db.query(Image).filter(Image.id == image_id).first()
    if image:
        return StreamingResponse(io.BytesIO(image.image_data), media_type="image/jpg")
    raise HTTPException(status_code=404, detail="Image not found")
