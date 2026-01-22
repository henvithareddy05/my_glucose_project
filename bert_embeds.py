# bert_embeds.py
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
import os

PREPROCESSED_CSV = r"C:\Users\henvitha\Desktop\my_glucose_project\data\preprocessed.csv"
EMBED_FILE = r"C:\Users\henvitha\Desktop\my_glucose_project\models\bert_embeds.npy"

def generate_bert_embeddings():
    df = pd.read_csv(PREPROCESSED_CSV)

    # Initialize BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Ensure text column exists
    text_col = 'notes'
    if text_col not in df.columns:
        df[text_col] = ""
    texts = df[text_col].fillna("")

    # Function to get embedding
    def get_embed(text):
        inputs = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        inputs = {k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:,0,:].cpu().numpy().squeeze()

    embeds = [get_embed(t) for t in texts]
    embeds = np.stack(embeds)

    os.makedirs(os.path.dirname(EMBED_FILE), exist_ok=True)
    np.save(EMBED_FILE, embeds)
    print(f"Saved BERT embeddings to '{EMBED_FILE}'")
    return embeds

if __name__ == "__main__":
    generate_bert_embeddings()
