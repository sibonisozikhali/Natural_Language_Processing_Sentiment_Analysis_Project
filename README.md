# Natural_Language_Processing_Sentiment_Analysis_Project

# Sentiment Analysis of Amazon Product Reviews (spaCy + TextBlob)

A Python NLP project that analyses the sentiment of Amazon consumer product reviews. It cleans the review text, scores each review for polarity and subjectivity, assigns a Positive / Negative / Neutral label, and compares how similar two reviews are using spaCy word vectors.

## What It Does

1. **Loads** the Datafiniti Amazon Consumer Reviews dataset with `pandas`.
2. **Preprocesses** the `reviews.text` column:
   - drops rows with missing review text
   - strips whitespace and lower-cases the text
   - removes stop words with spaCy (kept in a separate column)
3. **Scores sentiment** using `spacytextblob`, which adds TextBlob's analyser to the spaCy pipeline:
   - **Polarity:** from -1.0 (very negative) to 1.0 (very positive)
   - **Subjectivity:** from 0.0 (objective) to 1.0 (subjective)
   - **Label:** Positive (polarity > 0.05), Negative (polarity < -0.05), otherwise Neutral
4. **Summarises results:** label distribution across the dataset, and average polarity per star rating (`reviews.rating`).
5. **Tests** the `predict_sentiment()` function on three sample reviews.
6. **Compares similarity** between two reviews from the dataset using `en_core_web_md` word vectors.
7. **Saves** the labelled results to `sentiment_analysis_results.csv`.

## Project Structure

```
.
├── sentiment_analysis_1.py                                           # Main script
├── Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv   # Dataset (see note below)
├── sentiment_analysis_results.csv                                    # Output (generated on run)
└── README.md
```

## Requirements

- Python 3.8 or higher
- [pandas](https://pandas.pydata.org/)
- [spaCy](https://spacy.io/)
- [spacytextblob](https://github.com/SamEdwardes/spaCyTextBlob) (installs TextBlob as a dependency)
- spaCy's `en_core_web_md` English model

## Installation

```bash
# 1. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install pandas spacy spacytextblob

# 3. Download the spaCy model and TextBlob corpora
python -m spacy download en_core_web_md
python -m textblob.download_corpora
```

## Dataset

The script uses the **Datafiniti Amazon Consumer Reviews of Amazon Products** dataset (May 2019 version), which contains product reviews with fields such as review text, rating, and product details.

- The script expects the file to be named exactly `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv` and to sit in the same folder as the script. If your download has a different name (for example a `_1` suffix), rename it or update `DATA_PATH` in the script.
- The review text is read from the `reviews.text` column and star ratings from `reviews.rating`.

## Usage

```bash
python sentiment_analysis_1.py
```

The script prints its progress to the console and writes `sentiment_analysis_results.csv`.

### Output file

`sentiment_analysis_results.csv` contains one row per review with these columns:

| Column | Description |
|--------|-------------|
| `reviews.text` | Original review text |
| `polarity` | Sentiment score from -1.0 to 1.0 |
| `subjectivity` | Subjectivity score from 0.0 to 1.0 |
| `sentiment_label` | `Positive`, `Negative` or `Neutral` |

### Using the prediction function

`predict_sentiment()` scores a single review and returns a dictionary:

```python
predict_sentiment("This product is absolutely amazing, I love it and would buy it again!")
# {'polarity': ..., 'subjectivity': ..., 'label': 'Positive'}
```

## Design Notes

- **Stop words are removed into a separate column, but sentiment is scored on the cleaned text with stop words kept.** spaCy's default stop-word list includes negations like "not" and "no", and removing them can flip the apparent meaning of a review.
- **Batch processing:** `nlp.pipe()` is used instead of calling `nlp()` row by row, which is considerably faster on a large dataset.
- **Neutral dead-band:** polarity scores between -0.05 and 0.05 are labelled Neutral, because real text rarely scores exactly 0.
- **Similarity** is calculated from averaged word vectors in `en_core_web_md`, so it reflects overall topical closeness rather than exact meaning.

## Author

**Siboniso Zikhali**

## Acknowledgements

- Dataset: [Datafiniti](https://datafiniti.co/) Amazon Consumer Reviews of Amazon Products
- Libraries: [spaCy](https://spacy.io/), [spaCyTextBlob](https://github.com/SamEdwardes/spaCyTextBlob), [TextBlob](https://textblob.readthedocs.io/), [pandas](https://pandas.pydata.org/)
