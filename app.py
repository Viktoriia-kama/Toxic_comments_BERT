import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import zipfile
import os
import numpy as np

# Задайте шлях до моделі, токенайзера і датасету
MODEL_PATH = r"./model"
TOKENIZER_PATH = r'./model'
DATASET_PATH = r'./data/train.csv.zip'

# Завантаження моделі та токенайзера
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Завантаження датасету
# @st.cache
@st.cache_data
def load_dataset(path):
    return pd.read_csv(path,compression='zip')

train_df = load_dataset(DATASET_PATH)
# df = pd.read_csv(DATASET_PATH,compression='zip')

# Визначення міток
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Функція для передбачення
def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.sigmoid(logits)
    return probabilities.numpy().flatten()

# Streamlit інтерфейс
st.title('Text Classification with BERT')

# Ініціалізація стану коментаря
if 'comment_text' not in st.session_state:
    st.session_state.comment_text = ""

# Кнопка для вибору випадкового коментаря
if 'train_df' in locals():
    if st.button('Select Random Comment'):
        sample_text = train_df['comment_text'].sample(n=1).values[0]
        st.session_state.comment_text = sample_text

# Текстове поле для введення або відображення коментаря
comment_text = st.text_area('Comment Text', st.session_state.comment_text, height=100)

# Кнопка для передбачення
if st.button('Predict'):
    predictions = predict(comment_text, model, tokenizer)
    binary_predictions = (predictions > 0.5).astype(int)

    # Відображення результатів у таблиці
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    results = {label: [f"{pred:.2f}", binary] for label, pred, binary in zip(labels, predictions, binary_predictions)}

    st.table(pd.DataFrame(results, index=['Probability', 'Binary Prediction']).T)

else:
    st.write("Please load the dataset to proceed.")
    
# Запуск Streamlit додатка
if __name__ == "__main__":
    st.write("Run the script with `streamlit run your_script.py`")
