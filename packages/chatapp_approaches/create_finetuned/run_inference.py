import google.generativeai as genai

#change to read queries from test.csv
name = f'sr-medical-db'
query="What options are available for treating hearing loss?"

completion = genai.generate_text(model=f'tunedModels/{name}',
                                prompt=query)
print(completion.result)