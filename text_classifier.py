# packages required for the first step which is text cleaning
import re
import nltk
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk import pos_tag
#--------------------------------------
import pandas as pd
#--------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
#--------------------------------------
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
#--------------------------------------
import numpy as np
import random



random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# nltk download 

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# prepare stop words and lemmatizer
stop_words = set(stopwords.words('english'))
NEGATORS = {"not", "no", "never", "n't"}
stop_words = stop_words.difference(NEGATORS)
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

CONTRACTIONS = {
    "don't": "do not", "didn't": "did not", "doesn't": "does not",
    "can't": "can not", "won't": "will not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "shouldn't": "should not", "wouldn't": "would not", "couldn't": "could not",
    "mightn't": "might not", "mustn't": "must not", "ain't": "is not"
}
def expand_contractions(s: str) -> str:
    s = s.lower()
    for c, e in CONTRACTIONS.items():
        s = s.replace(c, e)
    return s


def clean_text(text):
    # expand contractions
    text = expand_contractions(text)
    # lowercase the text
    text = text.lower()

    # remove punctuation and non alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize into words
    tokens = word_tokenize(text)

    # Remove stopwords 
    cleaned_tokens = [word for word in tokens if word not in stop_words]

    stems  = [stemmer.stem(t) for t in cleaned_tokens]
    return stems


# load data 
def load_dataset(csv_path="feedback.csv"):
    df = pd.read_csv(csv_path)
    # keep only rows that have both text and label
    df = df.dropna(subset=["text", "label"])
    # enforce expected types
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


# vectorizer
def build_vectorizer():
    # Use our clean_text function as the tokenizer
    return CountVectorizer(
        tokenizer=clean_text,
        binary=True,              
        ngram_range=(1, 2)        
    )




# Split raw texts,  build BoW on train only
def split_and_vectorize(df, test_size=0.2, random_state=42):
    # 1) Split RAW TEXTS + labels (preserve class balance)
    X_text = df["text"]
    y = df["label"]
    X_text_train, X_text_test, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # keeps 0/1 ratio similar in both splits
    )

    # 2) Fit vectorizer on TRAIN TEXTS ONLY (avoid leakage)
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_text_train)

    # 3) Transform TEST TEXTS using the same fitted vectorizer
    X_test = vectorizer.transform(X_text_test)

    return X_train, X_test, y_train.values, y_test.values, vectorizer



# model build
def build_model(input_dim: int) -> Sequential:
    model = Sequential([
        Input(shape=(input_dim,)),      
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# train model
def train_model(model: Sequential, X_train, y_train):
    # Keras expects dense arrays
    X_train_dense = X_train.toarray()
    history = model.fit(
        X_train_dense,
        y_train,
        epochs=20,
        batch_size=4,
        validation_split=0.2,

    )
    return history

# evaluate model
def evaluate_model(model: Sequential, X_test, y_test):
    X_test_dense = X_test.toarray()
    loss, acc = model.evaluate(X_test_dense, y_test, verbose=0)
    print(f"Test accuracy: {acc:.3f}")
    return acc


# predict text 
def predict_text(text: str, vectorizer, model, threshold: float = 0.5, band: float = 0.0):
    X = vectorizer.transform([text]).toarray()
    prob = float(model.predict(X, verbose=0)[0][0])
    if abs(prob - threshold) <= band:
        return {"prob": prob, "label": "Unsure"}
    return {"prob": prob, "label": "Positive" if prob >= threshold else "Negative"}




if __name__ == "__main__":
    # 1) Load + split + vectorize
    df = load_dataset("feedback.csv")
    X_train, X_test, y_train, y_test, vectorizer = split_and_vectorize(df)

    print("Train shape:", X_train.shape, " Test shape:", X_test.shape)
    print("Vocab size (train-only):", len(vectorizer.get_feature_names_out()))
    print("Train label counts:", {0: int((y_train==0).sum()), 1: int((y_train==1).sum())})
    print("Test  label counts:", {0: int((y_test==0).sum()), 1: int((y_test==1).sum())})

    # 2) Build model
    model = build_model(input_dim=X_train.shape[1])
    model.summary()

    # 3) Train
    _ = train_model(model, X_train, y_train)

    # 4) Evaluate
    _ = evaluate_model(model, X_test, y_test)

 # --- Interactive prediction loop ---
    print("\nType feedback to classify (or 'quit' to exit):")
    try:
        while True:
            user_input = input("> ").strip()
            if user_input.lower() in {"q", "quit", "exit"}:
                break
            if user_input.lower() in {"", " "}:
                print("This is an empty phrase please enter valid feedback")
                continue
            out = predict_text(user_input, vectorizer, model)
            if out["prob"] is None:
                print(f"Prediction: {out['label']}")
            else:
                print(f"Prediction: {out['label']}  (prob={out['prob']:.2f})")
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")