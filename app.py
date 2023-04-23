from flask import Flask, request, jsonify
import openai
import os
from PyPDF2 import PdfReader
import numpy as np
import pandas as pd
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import pickle

app = Flask(__name__)

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')


module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
embed = hub.load(module_url)

job_desc_embeddings = []
job_kw_embeddings = []

with open('job_desc_embeddings_pickle.pkl', 'rb') as f:
    job_desc_embeddings = pickle.load(f)

with open('job_kw_embeddings_pickle.pkl', 'rb') as f:
    job_kw_embeddings = pickle.load(f)

def preprocess_text(text):
    text = text.lower().strip()
    text = " ".join(text.split())
    return text

def compute_embeddings(text):
    return embed(text)



jobs_data = pd.read_csv("naukriAll1.csv")

jobs_data["text"] = jobs_data["jobDescription"].apply(preprocess_text)

jobs_data["keywords"] = jobs_data["tagsAndSkills"].apply(preprocess_text)


def recommend_jobs(user_profile):
    user_profile_text = preprocess_text(user_profile)
    user_profile_embedding = compute_embeddings([user_profile_text]).numpy()[0]
    desc_similarity_scores = cosine_similarity(user_profile_embedding.reshape(1, -1), job_desc_embeddings)
    kw_similarity_scores = cosine_similarity(user_profile_embedding.reshape(1, -1), job_kw_embeddings)
    combined_similarity_scores = 0.8*desc_similarity_scores + 0.2*kw_similarity_scores
    top_indices = np.argsort(-combined_similarity_scores, axis=1).flatten()
    df = jobs_data.iloc[top_indices][["title", "companyName", "jdURL", "minExp", "maxExp", "salary", "location"]]
    return df.sort_values(by=['minExp'])


@app.route('/ping/', methods=['GET'])
def welcome():
    return "Hello World! the flask app is working"

@app.route('/generate-recommendations/', methods=['GET'])
def summarize_pdf():
    
    # Get the uploaded PDF file
    pdf_file = request.files['file']
    
    # Read the PDF file and extract its text content
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for i in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[i].extract_text()

    prompt = "Summarize the following text in less than 70 words:\n" + text,
    
    # Generate a summary using OpenAI API
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=140
    )
    summary = response.choices[0].text.strip()
    
    df = recommend_jobs(summary)
    return jsonify(df.to_json(orient='records'))



if __name__ == '__main__':
    app.run()