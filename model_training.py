import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import DistilBertTokenizer


df = pd.read_csv("smash_society_dataset.csv")



etiketler = df['label'].unique()
etiket_haritasi = {label: i for i, label in enumerate(etiketler)}
df['label_num'] = df['label'].map(etiket_haritasi)


print("Etiket Haritası:", etiket_haritasi)

# Eğitim (%80) ve Test (%20) 
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Pandas formatından Hugging Face Dataset formatına çeviriyoruz
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


# Hugging Face'in  DistilBERT modelinin sözlüğü
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_et(ornekler):
    return tokenizer(ornekler['text'], padding="max_length", truncation=True)

# Tokenizer'ı tüm veri setimize uyguluyoruz
train_dataset = train_dataset.map(tokenize_et, batched=True)
test_dataset = test_dataset.map(tokenize_et, batched=True)


train_dataset = train_dataset.rename_column("label_num", "labels")
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

test_dataset = test_dataset.rename_column("label_num", "labels")
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

print("Tokenizasyon tamamlandı!")



# modeli yüklüyoruz (DistilBERT).
num_labels = len(etiket_haritasi)
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

#  accuracy ölçüyoruz.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# training
training_args = TrainingArguments(
    output_dir='./smash_society_model_sonuclar', 
    num_train_epochs=3,              
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=8,
    eval_strategy="epoch",          
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

 
print("Eğitim başlıyor...")
trainer.train()


trainer.save_model("./smash_society_final_model")
tokenizer.save_pretrained("./smash_society_final_model")

print("!!!Smash Society Müşteri Asistanı başarıyla eğitildi ve kaydedildi!!!")