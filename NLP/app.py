import speech_recognition as sr
from textblob import TextBlob
from datetime import datetime
import os
from langdetect import detect
import nltk
from nltk.corpus import stopwords
from collections import Counter
import string

# Download stopwords on first run
nltk.download('stopwords')

# Supported languages
LANGUAGES = {
    "1": ("English", "en-US"),
    "2": ("Hindi", "hi-IN"),
    "3": ("French", "fr-FR"),
    "4": ("Spanish", "es-ES"),
    "5": ("German", "de-DE"),
    "6": ("Kannada", "kn-IN"),
}

def choose_language():
    print("\n Choose language:")
    for key, (lang, code) in LANGUAGES.items():
        print(f"{key}. {lang} ({code})")
    choice = input("Enter your choice: ").strip()
    return LANGUAGES.get(choice, LANGUAGES["1"])[1]  # Default: English

def detect_intent(text):
    text = text.lower()
    if any(kw in text for kw in ["weather", "temperature", "rain", "sunny", "cloudy", "forecast"]):
        return "Intent: Weather Inquiry"
    elif any(kw in text for kw in ["time", "date", "day", "month", "year", "clock"]):
        return "Intent: Time/Date Inquiry"
    elif any(kw in text for kw in ["play", "music", "song", "listen", "radio", "tune"]):
        return "Intent: Music Command"
    elif any(kw in text for kw in ["hello", "hi", "how are you", "greetings", "hey"]):
        return "Intent: Greeting"
    elif any(kw in text for kw in ["joke", "funny", "laugh", "humor"]):
        return "Intent: Tell a Joke"
    elif any(kw in text for kw in ["news", "headlines", "update", "current affairs"]):
        return "Intent: News Request"
    elif any(kw in text for kw in ["help", "support", "assist", "problem"]):
        return "Intent: Help Request"
    elif any(kw in text for kw in ["thank", "thanks", "appreciate"]):
        return "Intent: Gratitude"
    elif any(kw in text for kw in ["bye", "goodbye", "see you", "later"]):
        return "Intent: Goodbye"
    else:
        return "Intent: General Statement"

def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
    most_common = Counter(filtered_words).most_common(5)
    return [word for word, count in most_common]

def log_to_file(text, sentiment, subjectivity, intent, lang_code, keywords):
    os.makedirs("logs", exist_ok=True)
    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    filename = f"logs/speech_log_{now}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Timestamp: {now}\n")
        f.write(f"Language: {lang_code}\n")
        f.write(f"Transcription: {text}\n")
        f.write(f"Sentiment Polarity: {sentiment:.2f}\n")
        f.write(f"Sentiment Subjectivity: {subjectivity:.2f}\n")
        f.write(f"{intent}\n")
        f.write(f"Keywords: {', '.join(keywords)}\n")
    print(f"✅ Logged to {filename}")

recognizer = sr.Recognizer()
lang_code = choose_language()

with sr.Microphone() as source:
    print(f"\nSpeak now in {lang_code}...")
    recognizer.adjust_for_ambient_noise(source)
    audio = recognizer.listen(source)

    try:
        # Get full response with confidence
        response = recognizer.recognize_google(audio, language=lang_code, show_all=True)
        
        if not response or "alternative" not in response:
            raise sr.UnknownValueError()

        # Pick top result
        top_result = response["alternative"][0]
        text = top_result.get("transcript", "")
        confidence = top_result.get("confidence", None)  # May not always be present
        
        print("\nTranscription:", text)
        if confidence:
            print(f"Confidence Score: {confidence:.2f}")
        else:
            print("Confidence Score: Not available")

        # Sentiment (only for English)
        if lang_code == "en-US":
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            print(f"Sentiment Polarity: {sentiment:.2f}")
            print(f"Sentiment Subjectivity: {subjectivity:.2f}")
        else:
            sentiment = 0
            subjectivity = 0
            print("📌 Sentiment analysis skipped for this language.")

        # Intent detection only for English
        if lang_code == "en-US":
            intent = detect_intent(text)
            print(intent)
        else:
            intent = "Intent detection skipped."
            print(intent)

        # Keywords extraction (only English)
        keywords = extract_keywords(text) if lang_code == "en-US" else []
        if keywords:
            print(f"Keywords: {', '.join(keywords)}")

        # Language detection (confirm detected language)
        detected_lang = detect(text)
        print(f"Detected Language (langdetect): {detected_lang}")

        log_to_file(text, sentiment, subjectivity, intent, lang_code, keywords)

    except sr.UnknownValueError:
        print("❌ Could not understand the audio.")
    except sr.RequestError as e:
        print(f"⚠️ API request failed: {e}")
