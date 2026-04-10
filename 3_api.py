from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Smash Society AI API")

# React arayüzümüzün bu API'ye bağlıyoruz
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Eğittiğimiz modeli yüklüyoruz
MODEL_PATH = "./smash_society_final_model"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

etiketler = {
    0: 'Location / Address Request📍', 
    1: 'Complaint Creation Request ⚠️', 
    2: 'Review/Feedback Submission Request ⭐', 
    3: 'Menu Information 🍔', 
    4: 'Online Order 🛵', 
    5: 'Reservation 📅'
}

# React'ten gelecek verinin formatı
class MesajIstegi(BaseModel):
    text: str

@app.post("/tahmin")
def niyet_tahmin_et(istek: MesajIstegi):
    
    inputs = tokenizer(istek.text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class_id = outputs.logits.argmax().item()
        
    return {"niyet": etiketler[predicted_class_id]}

print(" API Başlatılıyor... React için hazır!")