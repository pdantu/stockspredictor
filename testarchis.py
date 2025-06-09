import pandas as pd
df = pd.read_csv('sentiment_scores.csv')
df = df.sort_values(by='SentimentScore', ascending=False).reset_index(drop=True)
df.to_csv('sentiment_scores_sorted.csv', index=False)