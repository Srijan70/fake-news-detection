import joblib
import re
import os
import urllib.request
from html.parser import HTMLParser
import nltk
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Ensure NLTK packages are downloaded
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

try:
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
except Exception as e:
    print(f"Failed to load NLTK VADER sentiment analyzer: {e}. Falling back to lexicon-based sentiment analysis.")
    sia = None

app = FastAPI(title="Fake News Detection API")

# Add CORS middleware so the frontend can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and vectorizer
try:
    model = joblib.load("fakenews_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    model = None
    vectorizer = None

stop_words = set(stopwords.words('english'))

def clean(text):
    text = re.sub(r'\W', ' ', str(text).lower())
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

# Buzzwords list for clickbait detection
CLICKBAIT_WORDS = {
    "shocking", "unbelievable", "you won't believe", "secret", "exposed", 
    "click here", "miracle", "leak", "breaking", "mind-blowing", "viral", 
    "obliterates", "destroys", "insane", "confession", "revealed", "hiding",
    "hidden", "conspiracy", "scandal", "truth about", "what happens next"
}

POSITIVE_WORDS = {"great", "good", "excellent", "success", "achievement", "victory", "safe", "healthy", "trustworthy", "approved", "positive", "helpful", "progress", "win", "benefit", "boost"}
NEGATIVE_WORDS = {"bad", "worst", "fail", "failed", "danger", "dangerous", "crisis", "threat", "risk", "warn", "warning", "expose", "exposed", "fake", "hoax", "conspiracy", "scandal", "lie", "liar", "fraud", "scam", "arrest", "illegal", "death", "kill", "murder", "disaster", "ruin", "destroy"}

class NewsRequest(BaseModel):
    text: str

class TextStats(BaseModel):
    word_count: int
    char_count: int
    read_time: int  # minutes
    clickbait_score: int
    sentiment: str
    sentiment_score: float

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    stats: TextStats

# Custom HTMLParser for extracting plain text from web pages
class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.current_tag = []
        self.ignore_tags = {'script', 'style', 'header', 'footer', 'nav', 'aside', 'noscript', 'head', 'iframe', 'button', 'form'}

    def handle_starttag(self, tag, attrs):
        self.current_tag.append(tag.lower())

    def handle_endtag(self, tag):
        if self.current_tag:
            self.current_tag.pop()

    def handle_data(self, data):
        if not any(t in self.ignore_tags for t in self.current_tag):
            cleaned = data.strip()
            if cleaned:
                if 'body' in self.current_tag:
                    self.text_parts.append(cleaned)
                elif not self.current_tag:  # Fallback outside nested tags
                    self.text_parts.append(cleaned)

    def get_text(self):
        return "\n\n".join(self.text_parts)

class UrlRequest(BaseModel):
    url: str

class UrlResponse(BaseModel):
    title: str
    text: str

@app.post("/analyze-url", response_model=UrlResponse)
def analyze_url(request: UrlRequest):
    url = request.url
    if not url.startswith("http://") and not url.startswith("https://"):
        raise HTTPException(status_code=400, detail="Invalid URL format. Must start with http:// or https://")
    
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
        parser = TextExtractor()
        parser.feed(html)
        extracted_text = parser.get_text()
        
        # Extract title using regex
        title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else "Extracted Article"
        
        if not extracted_text.strip():
            raise HTTPException(status_code=422, detail="Failed to extract readable text content from the URL.")
            
        return {"title": title, "text": extracted_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch or parse URL: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: NewsRequest):
    if not model or not vectorizer:
        return {
            "prediction": "Error",
            "confidence": 0.0,
            "stats": {
                "word_count": 0,
                "char_count": 0,
                "read_time": 0,
                "clickbait_score": 0,
                "sentiment": "Neutral",
                "sentiment_score": 0.0
            }
        }
    
    raw_text = request.text
    cleaned_text = clean(raw_text)
    vectorized_text = vectorizer.transform([cleaned_text])
    
    prediction = model.predict(vectorized_text)[0]
    confidence_probs = model.predict_proba(vectorized_text)[0]
    
    # prediction 1 = Real, 0 = Fake
    label = "REAL" if prediction == 1 else "FAKE"
    score = round(max(confidence_probs) * 100, 2)
    
    # Calculate stats
    words = raw_text.split()
    word_count = len(words)
    char_count = len(raw_text)
    read_time = max(1, round(word_count / 200))
    
    # Clickbait scoring
    lower_text = raw_text.lower()
    buzz_count = sum(1 for word in CLICKBAIT_WORDS if word in lower_text)
    caps_words = len([w for w in words if w.isupper() and len(w) > 1])
    caps_ratio = (caps_words / word_count) if word_count > 0 else 0
    excl_count = raw_text.count("!")
    
    clickbait_score = buzz_count * 15
    if word_count > 5:
        clickbait_score += int(caps_ratio * 100)
    clickbait_score += excl_count * 5
    clickbait_score = min(100, max(0, clickbait_score))
    
    # Sentiment scoring
    sentiment_label = "Neutral"
    sentiment_score = 0.0
    
    if sia:
        try:
            vader_res = sia.polarity_scores(raw_text)
            sentiment_score = round(vader_res['compound'], 2)
            if sentiment_score >= 0.05:
                sentiment_label = "Positive"
            elif sentiment_score <= -0.05:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
        except Exception:
            sia_failed = True
        else:
            sia_failed = False
    else:
        sia_failed = True
        
    if sia_failed:
        # Fallback to simple lexicon matching
        pos_count = sum(1 for w in words if w.lower().strip(",.?!:;\"'") in POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w.lower().strip(",.?!:;\"'") in NEGATIVE_WORDS)
        total_s = pos_count + neg_count
        if total_s > 0:
            sentiment_score = round((pos_count - neg_count) / total_s, 2)
            if sentiment_score >= 0.1:
                sentiment_label = "Positive"
            elif sentiment_score <= -0.1:
                sentiment_label = "Negative"
        else:
            sentiment_score = 0.0
            sentiment_label = "Neutral"
            
    stats_data = {
        "word_count": word_count,
        "char_count": char_count,
        "read_time": read_time,
        "clickbait_score": clickbait_score,
        "sentiment": sentiment_label,
        "sentiment_score": sentiment_score
    }
    
    return {
        "prediction": label,
        "confidence": score,
        "stats": stats_data
    }

@app.get("/model-info")
def get_model_info():
    return {
        "algorithm": "Logistic Regression Classifier",
        "features": 5000,
        "accuracy": 98.7,  # Training classification report approximation
        "test_split": "20%",
        "vectorizer": "TF-IDF Vectorizer",
        "class_balancing": "SMOTE (Synthetic Minority Over-sampling Technique)"
    }

@app.get("/model-info/confusion-matrix")
def get_confusion_matrix():
    if os.path.exists("confusion_matrix.png"):
        return FileResponse("confusion_matrix.png", media_type="image/png")
    raise HTTPException(status_code=404, detail="Confusion matrix image not found")

@app.get("/")
def read_root():
    return {"status": "Model API is running", "endpoints": ["/predict", "/analyze-url", "/model-info", "/model-info/confusion-matrix"]}
