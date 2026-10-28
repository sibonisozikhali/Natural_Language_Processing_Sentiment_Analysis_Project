# ---------------------------------------------------------------------------
#                     Practical Task: NLP Applications
# ---------------------------------------------------------------------------
# Please save Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv
# on the same folder as the python script before running the python script.
# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# ---------------------------------------------------------------------------
# 2. Load the spaCy model and add the sentiment pipeline component
# ---------------------------------------------------------------------------
# en_core_web_md is a mid-sized English pipeline that includes word vectors,
# which is what powers the .similarity() comparisons used later on.
nlp = spacy.load("en_core_web_md")

# spacytextblob wraps TextBlob's sentiment analyser and attaches it to the
# spaCy pipeline, so every Doc we create gets a `._.blob` (and `._.polarity`,
# `._.subjectivity`) extension automatically.
nlp.add_pipe("spacytextblob")

# ---------------------------------------------------------------------------
# 3. Load the dataset
# ---------------------------------------------------------------------------
DATA_PATH = "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"

# low_memory=False avoids pandas' dtype-guessing warnings on this wide,
# mixed-type CSV (24 columns of varying types).
dataframe = pd.read_csv(DATA_PATH, low_memory=False)

print(f"Loaded {len(dataframe)} rows and {len(dataframe.columns)} columns.")

# ---------------------------------------------------------------------------
# 4. Select the review text column
# ---------------------------------------------------------------------------
# "reviews.text" is the column that holds the free-text product review, which
# is the feature we will run sentiment analysis on.
reviews_data = dataframe["reviews.text"]

# ---------------------------------------------------------------------------
# 5. Preprocess the text data
# ---------------------------------------------------------------------------
# 5a. Drop rows where the review text itself is missing, since there is
#     nothing to analyse for those rows.
clean_data = dataframe.dropna(subset=["reviews.text"])


def clean_text(text: str) -> str:
    """
    Perform basic text cleaning before sentiment scoring:
      - strip() removes leading/trailing whitespace
      - lower() normalises case so "Great" and "great" are treated the same
      - str() guards against any non-string values slipping through
    """
    return str(text).strip().lower()


# Apply the basic cleaning to every review and store the result in a new
# column so we keep the original text alongside the cleaned version.
clean_data = clean_data.copy()
clean_data["clean_text"] = clean_data["reviews.text"].apply(clean_text)


def remove_stopwords(text: str) -> str:
    """
    Tokenise `text` with spaCy and drop any token that spaCy flags as a
    stop word (token.is_stop). Stop words such as "the", "is", or "of"
    carry little sentiment-bearing meaning, so filtering them out keeps
    the remaining tokens focused on the words that actually signal
    opinion/sentiment.
    """
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if not token.is_stop]
    return " ".join(filtered_tokens)


# We keep both versions of the text -- `clean_text` (cased-normalised,
# stop words included) is what we feed to TextBlob/spaCy for sentiment,
# since removing stop words can occasionally distort polarity for short
# phrases (e.g. negations like "not"/"no" are stop words in spaCy's default
# list). `no_stopwords_text` is kept separately to satisfy the assignment's
# preprocessing requirement and for any downstream keyword-style analysis.
clean_data[
    "no_stopwords_text"] = clean_data["clean_text"].apply(remove_stopwords)

print("Preprocessing complete. Example of cleaned text:")
print(clean_data[["reviews.text", "clean_text", "no_stopwords_text"]].head(2))

# ---------------------------------------------------------------------------
# 6. Function for sentiment analysis
# ---------------------------------------------------------------------------


def predict_sentiment(review_text: str) -> dict:
    """
    Take a single product review (raw string) and return its predicted
    sentiment as a dictionary containing:
      - polarity: float in [-1.0, 1.0]. 1 = very positive, -1 = very
        negative, 0 = neutral.
      - subjectivity: float in [0.0, 1.0] from the underlying TextBlob
        sentiment tuple (0 = very objective/factual, 1 = very subjective/
        opinion-based).
      - label: a human-readable "Positive" / "Negative" / "Neutral" tag
        derived from the polarity score.

    A small dead-band around zero (-0.05 to 0.05) is used to classify a
    review as "Neutral", since real-world polarity scores are rarely
    exactly 0 even for genuinely neutral text.
    """
    text = clean_text(review_text)
    doc = nlp(text)

    polarity = doc._.blob.polarity
    subjectivity = doc._.blob.subjectivity

    if polarity > 0.05:
        label = "Positive"
    elif polarity < -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {"polarity": polarity, "subjectivity": subjectivity, "label": label}


# ---------------------------------------------------------------------------
# 7. Apply the sentiment function across the (cleaned) dataset
# ---------------------------------------------------------------------------
# Running spaCy + TextBlob over every review can be slow on large datasets,
# so this uses nlp.pipe() for efficient batch processing rather than calling
# nlp() once per row in a Python loop.
polarities = []
subjectivities = []
labels = []

for doc in nlp.pipe(clean_data["clean_text"], batch_size=64):
    polarities.append(doc._.blob.polarity)
    subjectivities.append(doc._.blob.subjectivity)
    if doc._.blob.polarity > 0.05:
        labels.append("Positive")
    elif doc._.blob.polarity < -0.05:
        labels.append("Negative")
    else:
        labels.append("Neutral")

clean_data["polarity"] = polarities
clean_data["subjectivity"] = subjectivities
clean_data["sentiment_label"] = labels

print("\nSentiment label distribution across the dataset:")
print(clean_data["sentiment_label"].value_counts())

print(
    "\nAverage polarity by star rating (if a 'reviews.rating' column exists):")
if "reviews.rating" in clean_data.columns:
    print(clean_data.groupby("reviews.rating")["polarity"].mean())

# ---------------------------------------------------------------------------
# 8. Test the model on sample product reviews
# ---------------------------------------------------------------------------
sample_reviews = [
    "This product is absolutely amazing, I love it and would buy it again!",
    "Terrible quality, it broke after two days and customer service was no "
    "help.",
    "It's an okay product, does what it says but nothing special.",
]

print("\n--- Testing predict_sentiment() on sample reviews ---")
for review in sample_reviews:
    result = predict_sentiment(review)
    print(f"\nReview: {review}")
    print(
        f"  -> Polarity: {result['polarity']:.3f} | "
        f"Subjectivity: {result['subjectivity']:.3f} | "
        f"Predicted sentiment: {result['label']}"
    )

# ---------------------------------------------------------------------------
# 9. Compare the similarity of two reviews from the dataset
# ---------------------------------------------------------------------------
# Select two reviews from the "reviews.text" column by index to demonstrate
# spaCy's word-vector-based similarity() function. Indices must stay within
# the bounds of the dataset (0 to len(dataframe) - 1).
my_review_of_choice = dataframe["reviews.text"][0]
another_review = dataframe["reviews.text"][1]

doc_a = nlp(str(my_review_of_choice))
doc_b = nlp(str(another_review))

similarity_score = doc_a.similarity(doc_b)

print("\n--- Comparing similarity of two sample reviews ---")
print(f"Review A: {my_review_of_choice}")
print(f"Review B: {another_review}")
print(f"Similarity score: {similarity_score:.3f}")

# ---------------------------------------------------------------------------
# 10. Save the processed/labelled dataset for use in the report
# ---------------------------------------------------------------------------
output_columns = [
    "reviews.text", "polarity", "subjectivity", "sentiment_label"]
clean_data[output_columns].to_csv(
    "sentiment_analysis_results.csv", index=False)
print("\nSaved labelled results to sentiment_analysis_results.csv")
