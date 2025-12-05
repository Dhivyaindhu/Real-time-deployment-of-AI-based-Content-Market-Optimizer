import textstat
from sentiment_analyzer import analyze_sentiment

def performance_metrics(text, sentiment_score):
    readability = textstat.flesch_reading_ease(text)

    engagement_score = (sentiment_score * 0.5) + (readability / 100 * 0.5)
    engagement_score = round(engagement_score * 100, 2)

    return {
        "readability": readability,
        "sentiment_strength": sentiment_score,
        "engagement_score": engagement_score
    }

def pick_best_version(generated_variations):
    highest_score = -1
    best = ""

    for text in generated_variations:
        _, sentiment_score = analyze_sentiment(text)
        metrics = performance_metrics(text, sentiment_score)
        if metrics["engagement_score"] > highest_score:
            highest_score = metrics["engagement_score"]
            best = text

    return best, highest_score
