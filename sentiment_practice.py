import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

print("Starting sentiment analysis...")

# Load the pre-trained DistilRoberta model for financial sentiment analysis
model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define the label mapping
labels = ["negative", "neutral", "positive"]

# Function to predict sentiment
def predict_sentiment(text):
    # Handle NaN or empty text
    if pd.isna(text) or text == "":
        return "error", 0.0
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
    
    sentiment = labels[predicted_class]
    confidence = predictions[0][predicted_class].item()
    return sentiment, confidence

# Read the CSV file containing stock tweets
# Lese alle Zeilen anstatt nur 100 für eine vollständige Analyse
print("Lese alle Tweets aus der CSV-Datei...")
df = pd.read_csv("stock_tweets.csv")

# Convert the date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Add month and year columns for time series analysis
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['YearMonth'] = df['Date'].dt.strftime('%Y-%m')

# Check the dataframe structure
print("CSV columns:", df.columns)
print(f"Insgesamt {len(df)} Tweets gefunden.")
print("Sample data:")
print(df.head(2))

# Add sentiment analysis columns to the dataframe
results = []

# Process tweets with a progress bar
for index, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing tweets"):
    try:
        # Use the 'Tweet' column that contains the tweet text
        text = row['Tweet']
        sentiment, confidence = predict_sentiment(text)
        results.append({
            "index": int(index),
            "date": row['Date'].strftime('%Y-%m-%d %H:%M:%S%z'),
            "text": text,
            "stock": row['Stock Name'],
            "company": row['Company Name'],
            "year": int(row['Year']),
            "month": int(row['Month']),
            "yearmonth": row['YearMonth'],
            "sentiment": sentiment,
            "confidence": float(confidence)  # Ensure JSON serialization works
        })
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        results.append({
            "index": int(index),
            "date": row['Date'].strftime('%Y-%m-%d %H:%M:%S%z') if pd.notna(row['Date']) else "N/A",
            "text": row['Tweet'] if 'Tweet' in row and pd.notna(row['Tweet']) else "N/A",
            "stock": row['Stock Name'] if 'Stock Name' in row and pd.notna(row['Stock Name']) else "N/A",
            "company": row['Company Name'] if 'Company Name' in row and pd.notna(row['Company Name']) else "N/A",
            "year": int(row['Year']) if pd.notna(row['Year']) else 0,
            "month": int(row['Month']) if pd.notna(row['Month']) else 0,
            "yearmonth": row['YearMonth'] if pd.notna(row['YearMonth']) else "N/A",
            "sentiment": "error",
            "confidence": 0.0
        })

# Convert results to DataFrame for analysis
results_df = pd.DataFrame(results)

# Display sentiment distribution
sentiment_counts = results_df['sentiment'].value_counts()
print("\nSentiment Distribution:")
print(sentiment_counts)
print(f"Positive: {sentiment_counts.get('positive', 0)} ({sentiment_counts.get('positive', 0)/len(results_df)*100:.1f}%)")
print(f"Neutral: {sentiment_counts.get('neutral', 0)} ({sentiment_counts.get('neutral', 0)/len(results_df)*100:.1f}%)")
print(f"Negative: {sentiment_counts.get('negative', 0)} ({sentiment_counts.get('negative', 0)/len(results_df)*100:.1f}%)")

# Analyze sentiment by stock
print("\nSentiment by Stock:")
stock_sentiment = results_df.groupby(['stock', 'sentiment']).size().unstack(fill_value=0)
print(stock_sentiment)

# Analyze sentiment by month for each stock
print("\nMonthly Sentiment Analysis:")
monthly_sentiment = results_df.groupby(['stock', 'yearmonth', 'sentiment']).size().unstack(fill_value=0)
print(monthly_sentiment.head(10))

# Calculate sentiment score for each stock and month
# Create a pivot table for easier calculations
pivot_df = results_df.groupby(['stock', 'yearmonth', 'sentiment']).size().reset_index(name='count')
pivot_table = pivot_df.pivot_table(
    index=['stock', 'yearmonth'], 
    columns='sentiment', 
    values='count',
    fill_value=0
).reset_index()

# Ensure all sentiment columns exist
for sentiment in labels:
    if sentiment not in pivot_table.columns:
        pivot_table[sentiment] = 0

# Calculate sentiment score: (positive - negative) / total
pivot_table['total'] = pivot_table['positive'] + pivot_table['negative'] + pivot_table['neutral']
pivot_table['sentiment_score'] = (pivot_table['positive'] - pivot_table['negative']) / pivot_table['total']

print("\nSentiment Score by Month and Stock:")
print(pivot_table[['stock', 'yearmonth', 'sentiment_score']].head(10))

# Save results to JSON and CSV files
with open("sentiment_analysis_results.json", "w") as json_file:
    json.dump(results, json_file, indent=2)
print("\nResults saved to sentiment_analysis_results.json")

# Export pivot table to CSV for time series analysis
pivot_table.to_csv("monthly_sentiment_data.csv", index=False)
print("Monthly sentiment data saved to monthly_sentiment_data.csv")

# Optional: Display some examples of each sentiment
print("\nExamples of analyzed tweets:")
for sentiment in labels:
    examples = results_df[results_df['sentiment'] == sentiment].head(2)
    if not examples.empty:
        print(f"\n{sentiment.upper()} examples:")
        for _, example in examples.iterrows():
            print(f"Stock: {example['stock']} ({example['company']})")
            print(f"Date: {example['date']}")
            print(f"Text: {example['text'][:100]}{'...' if len(example['text']) > 100 else ''}")
            print(f"Confidence: {example['confidence']:.4f}")

print("\nNow run time_series_sentiment.py to generate the time series visualizations.")

