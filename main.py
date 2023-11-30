from typing import List

from PIL import Image as PilImage
from fastapi import FastAPI, Request, UploadFile, File, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, LargeBinary, String
from sqlalchemy.ext.declarative import declarative_base
from database import SessionLocal
from fastapi.responses import HTMLResponse, StreamingResponse
import io
import models
from models import Image
from transformers import CLIPProcessor, CLIPModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 데이터베이스 모델 정의
Base = declarative_base()
Base = models.Base

# 테이블 생성
Base.metadata.create_all(bind=create_engine("sqlite:///images.sqlite3", connect_args={'check_same_thread': False}))
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
labels = ["human", "landscape", "animal", "food", "document", "something else"]

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/uploadfile")
async def create_upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        # 파일 데이터 읽기
        contents = await file.read()

        # 이미지 분류
        image = PilImage.open(io.BytesIO(contents))
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        best_label_idx = probs.argmax().item()

        # 분류명과 파일명 설정
        category = labels[best_label_idx]
        filename = f"{category}_{file.filename}"

        # 데이터베이스에 저장
        db_image = Image(image_data=contents, category=category)
        db.add(db_image)
        db.commit()
        db.refresh(db_image)

        return {"filename": file.filename, "content_type": file.content_type, "category": category, "id": db_image.id}
    except HTTPException as e:
        # 유효성 검사 오류를 기록하거나 출력합니다.
        print(e.detail)
        raise

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

@app.get("/human")
async def read_human(request: Request, db: Session = Depends(get_db)):
   # 데이터베이스에서 이미지 목록 가져오기
    images = db.query(Image).filter(Image.category == "human").all()
    return templates.TemplateResponse("human.html", {"request": request, "images": images})

@app.get("/animal")
async def read_animal(request: Request, db: Session = Depends(get_db)):
   # 데이터베이스에서 이미지 목록 가져오기
    images = db.query(Image).filter(Image.category == "animal").all()
    return templates.TemplateResponse("animal.html", {"request": request, "images": images})

@app.get("/food")
async def read_food(request: Request, db: Session = Depends(get_db)):
   # 데이터베이스에서 이미지 목록 가져오기
    images = db.query(Image).filter(Image.category == "food").all()
    return templates.TemplateResponse("food.html", {"request": request, "images": images})


@app.get("/document")
async def read_document(request: Request, db: Session = Depends(get_db)):
   # 데이터베이스에서 이미지 목록 가져오기
    images = db.query(Image).filter(Image.category == "document").all()
    return templates.TemplateResponse("document.html", {"request": request, "images": images})


@app.get("/landscape")
async def read_landscape(request: Request, db: Session = Depends(get_db)):
   # 데이터베이스에서 이미지 목록 가져오기
    images = db.query(Image).filter(Image.category == "landscape").all()
    return templates.TemplateResponse("landscape.html", {"request": request, "images": images})


@app.get("/something")
async def read_something(request: Request, db: Session = Depends(get_db)):
   # 데이터베이스에서 이미지 목록 가져오기
    images = db.query(Image).filter(Image.category == "something else").all()
    return templates.TemplateResponse("something.html", {"request": request, "images": images})
