from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re
import emoji
import nltk
import joblib
import string
from io import BytesIO
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from googleapiclient.discovery import build
import os


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
sgd_model = joblib.load("sgd_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


hyperlink_pattern = re.compile(
    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
)

# Request model
class YouTubeRequest(BaseModel):
    url: str
    api_key: str

# Preprocessing
def text_processing(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), "", text)
    text = re.sub("^a-zA-Z0-9$,", "", text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\W', ' ', text)
    emoji.demojize(text)
    text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])
    return text

# Comment fetcher
def get_comments(video_url: str, api_key: str) -> List[str]:
    watch_index = video_url.find("watch?v=")
    if watch_index == -1:
        raise ValueError("Invalid YouTube URL.")

    video_id = video_url[watch_index + len('watch?v='): watch_index + len('watch?v=') + 11]
    youtube = build('youtube', 'v3', developerKey=api_key)
    video_response = youtube.videos().list(part='snippet', id=video_id).execute()

    if not video_response['items']:
        raise ValueError("Invalid or inaccessible video ID.")

    uploader_channel_id = video_response['items'][0]['snippet']['channelId']
    comments = []
    nextPageToken = None

    while len(comments) < 1000:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            pageToken=nextPageToken
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            if comment.get('authorChannelId', {}).get('value') != uploader_channel_id:
                comment_text = comment['textDisplay']
                if not hyperlink_pattern.search(comment_text):
                    comments.append(comment_text)

        nextPageToken = response.get('nextPageToken')
        if not nextPageToken:
            break

    return comments

# Sentiment analysis
def analyze_sentiment(comments: List[str]):
    processed = [text_processing(c) for c in comments]
    X = vectorizer.transform(processed)
    preds = sgd_model.predict(X)
    return {
        "positive": int((preds == 1).sum()),
        "negative": int((preds == -1).sum()),
        "neutral": int((preds == 0).sum()),
        "total_comments": len(preds),
        "raw_predictions": preds.tolist()
    }


@app.post("/predict")
def predict(video: YouTubeRequest):
    try:
        comments = get_comments(video.url, video.api_key)
        return analyze_sentiment(comments)
    except Exception as e:
        return {"error": str(e)}


@app.post("/plot")
def plot_sentiment(video: YouTubeRequest):
    try:
        comments = get_comments(video.url, video.api_key)
        stats = analyze_sentiment(comments)

        fig, ax = plt.subplots()
        labels = ['Positive', 'Negative', 'Neutral']
        values = [stats['positive'], stats['negative'], stats['neutral']]
        ax.bar(labels, values, color=['green', 'red', 'gray'])
        ax.set_title('YouTube Comment Sentiment Distribution')
        ax.set_ylabel('Number of Comments')

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        return Response(content=buf.read(), media_type="image/png")
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
