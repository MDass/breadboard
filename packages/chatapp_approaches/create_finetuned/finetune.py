import pandas as pd
import json
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# genai.configure(api_key=os.getenv('PALM_KEY'))

df = pd.read_csv("../train.csv")
df = df[["Question", "Answer"]]

df = df.rename(columns={'Question': 'text_input', 'Answer': 'output'})

df = df[df['output'].str.len() <= 5000]

jsons = df.to_json(orient='records')
jsons = json.loads(jsons)

# Restrict the amount since len(examples) x epoch_count must be at maximum 250,000
jsons = jsons[:2500]

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