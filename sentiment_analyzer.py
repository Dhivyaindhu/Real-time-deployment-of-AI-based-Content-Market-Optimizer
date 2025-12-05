from transformers import pipeline

sentiment_model = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_model(text)[0]
    sentiment = result["label"]
    score = round(result["score"], 3)
    return sentiment, score
