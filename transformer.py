import pandas as pd
from transformers import pipeline
#from collections import Counter
import torch

df = pd.read_csv('reviews.csv')

# Subset the data to 2,000 random reviews
seed = 844
subset_df = df.sample(n=2000, random_state=seed) 

# Change the device in case using GPU 
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", batch_size=32, device=-1)

# Run sentiment analysis on the subset
texts = subset_df['text'].tolist()
results = sentiment_pipeline(texts, batch_size=32) 

# Extract sentiment labels and confidence scores
subset_df['Sentiment'] = [result['label'] for result in results]
subset_df['Confidence'] = [result['score'] for result in results]

subset_df.head()
