import pandas as pd
import json
import google.generativeai as genai


df = pd.read_csv("create_finetuned/train.csv")
df = df[["Question", "Answer"]]

df = df.rename(columns={'Question': 'text_input', 'Answer': 'output'})

jsons = df.to_json(orient='records')
jsons = json.loads(jsons)

base_model = "models/text-bison-001"

name = f'sr-medical-db'
operation = genai.create_tuned_model(
    source_model=base_model,
    training_data=jsons,
    id = name,
    epoch_count = 100,
    batch_size=4,
    learning_rate=0.001,
)