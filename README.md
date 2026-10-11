# IMDb Sentiment Analysis Using SVM

An NLP and machine learning project that classifies IMDb movie reviews as positive or negative using TF-IDF text features and Support Vector Machine (SVM) models.

## Project Overview

This project uses 50,000 IMDb movie reviews to build and evaluate sentiment classification models. It compares Linear, Polynomial, and RBF SVM approaches and examines model errors.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- NLTK
- Matplotlib and Seaborn
- TF-IDF Vectorization
- Linear SVM, Polynomial SVM, and RBF SVM

## Workflow

1. Load and explore the IMDb review dataset.
2. Clean and preprocess review text.
3. Convert text into numerical features using TF-IDF.
4. Train and tune SVM classification models.
5. Evaluate accuracy, precision, recall, F1-score, and ROC-AUC.
6. Compare model performance and inspect misclassified reviews.

## Results

The optimized Linear SVM achieved the following results on the 25,000-review test set:

| Metric | Score |
|---|---:|
| Accuracy | 87.996% |
| Precision | 87.56% |
| Recall | 88.576% |
| F1-score | 88.07% |
| ROC-AUC | 0.9516 |

The optimized Linear SVM outperformed the Polynomial and RBF SVM models in the reported comparison.

## Dataset

The project uses the IMDb Large Movie Review Dataset, containing 25,000 training reviews and 25,000 test reviews.

Dataset source: [Stanford AI Lab — IMDb Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

The dataset is downloaded separately and is not included in this repository.

## How to Run

1. Open `imdb_sentiment_analysis.ipynb` in Google Colab.
2. Run the notebook cells in order.
3. Install any required libraries and download the dataset when prompted.
4. Train and evaluate the models.

## Author

Mahaprasad Acharjee
