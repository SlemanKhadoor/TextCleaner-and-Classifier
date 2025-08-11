# install
pip install nltk pandas scikit-learn tensorflow
pip install tensorflow




## The goal of this project is to develop a sentiment classification system that can automatically determine whether a piece of customer feedback is Positive or Negative.
The system is built using Python, NLTK for text preprocessing, scikit-learn for feature extraction, and TensorFlow/Keras for machine learning model training.
This report explains the process step-by-step, from data preparation to model evaluation.

## Dataset
The dataset (feedback.csv) contains labeled feedback messages where:

text: the customer feedback string.

label: sentiment category — 1 for Positive, 0 for Negative.


## Preprocessing
Customer feedback text can contain noise, punctuation, and irrelevant words. Preprocessing ensures the model focuses only on meaningful content.

### Steps performed
#### Contraction Expansion

Converts short forms to their full meaning.
Example: "don't" → "do not", "can't" → "can not".
This helps preserve negations, which are critical for sentiment.

#### Lowercasing

Converts all text to lowercase for consistency.

#### Removing Non-Alphabetic Characters

Removes punctuation, numbers, and special characters using regex.

#### Tokenization

Splits text into individual words using nltk.word_tokenize.

#### Stopword Removal (with Negation Preservation)

Removes common words such as “the” or “is” that add little meaning.

#### Negators like "not", "no", "never" are kept to preserve sentiment context.

#### Stemming

Reduces words to their base form.
Example: "loving", "loved" → "love".


## Feature Extraction
The model cannot directly work with text, so we convert it into numeric form using Bag-of-Words (BoW) with scikit-learn’s CountVectorizer.

### Configuration
Tokenizer: Custom cleaner function from preprocessing step.

binary=True: Records only presence/absence of a word (not frequency).

ngram_range=(1,2): Captures both single words (unigrams) and two-word phrases (bigrams).
This is crucial to distinguish "good" (positive) from "not good" (negative).



## Data Splitting
The dataset is split into:

Training Set (80%): Used to train the model.

Test Set (20%): Used to evaluate the model.

Stratified splitting ensures both sets have a similar positive/negative ratio.

## Model Architecture
We use a simple feedforward neural network in TensorFlow/Keras.

Architecture:

Input Layer – Size = number of vocabulary features.

Dense Layer (16 neurons, ReLU) – Learns intermediate sentiment patterns.

Dense Output Layer (1 neuron, Sigmoid) – Produces probability between 0 and 1.

Compilation Settings:

Optimizer: Adam

Loss Function: Binary Crossentropy

Metric: Accuracy

## Training
Epochs: 20

Batch Size: 4

Validation Split: 20% of training set used for validation during training.

Training converts the sparse BoW matrix to a dense array for Keras compatibility.

## Evaluation
The model is evaluated on the test set after training.




## Prediction on New Data
The model includes an interactive loop where a user can type feedback, and the system predicts:

Positive if probability ≥ 0.5

Negative if probability < 0.5

Optionally, “Unsure” if probability is near 0.5 within a tolerance band.



