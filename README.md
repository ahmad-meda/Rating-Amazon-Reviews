# Rating Amazon Reviews

This project focuses on analyzing Amazon product reviews and predicting the associated ratings using Machine Learning and Transformer-based models.  
It includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, topic modeling, and Kaggle-style submission preparation.

---

## Project Overview

The goal of this project is to:
- Analyze customer sentiment using Amazon review data
- Train and evaluate multiple classification models
- Compare traditional Machine Learning models with Transformer models
- Perform topic modeling on positive and negative reviews
- Prepare predictions for Kaggle submission

---

## Tech Stack

- Python (Jupyter Notebook)
- Scikit-learn
- Pandas and NumPy
- Matplotlib and Seaborn
- NLTK (Natural Language Toolkit)
- Gensim (for LDA topic modeling)
- Hugging Face Transformers (ModernBERT, DeBERTa)
- PyTorch

---

## Project Structure

| Section | Description |
|:-------|:------------|
| Introduction | Project overview and dataset |
| Data Preprocessing | Cleaning and preparing review text |
| Exploratory Data Analysis (EDA) | Insights and visualizations |
| Feature Engineering | TF-IDF vectorization |
| Model Training | Random Forest, SVM, Logistic Regression, SGD, etc. |
| Transformer Fine-Tuning | ModernBERT and DeBERTa fine-tuning |
| Model Evaluation | Accuracy, loss analysis |
| Topic Modeling | LDA applied to 5-star and 1-star reviews |
| Submission Preparation | Kaggle-style output |

---

## Setup Instructions

### 1. Clone the Repository

`git clone https://github.com/ahmad-meda/Rating-Amazon-Reviews.git`  
`cd Rating-Amazon-Reviews`

### 2. Install Required Packages

`pip install -r requirements.txt`

If no `requirements.txt` exists:

`pip install pandas numpy matplotlib seaborn scikit-learn nltk gensim transformers torch`

### 3. Download Transformer Models

When running the Transformers part, `bert-base-uncased` and `microsoft/deberta-large` will automatically download from Hugging Face.

---

## Machine Learning Model Performance

### Model Performance Across Vectorizations

This plot compares three models — **SGD**, **Logistic Regression**, and **Multinomial Naive Bayes** — across different text vectorization techniques (Count Vectorizer, TF-IDF Unigram, Bigram, Trigram).

- SGD classifier consistently performs better across all vectorizations.
- Logistic Regression performs slightly lower than SGD.
- Naive Bayes works better with simpler vectorizations like CountVectorizer but falls behind with complex n-grams.

![image](https://github.com/user-attachments/assets/3ab0be68-9d2f-49d9-b641-436604a00e98)



---

### Model Performance After Hyperparameter Tuning (GridSearchCV)

This plot shows how performance improves after applying **GridSearchCV** for hyperparameter tuning.

- Models, especially Multinomial Naive Bayes, show significant performance improvements.
- SGD maintains its leading position.
- Careful hyperparameter tuning enhances model robustness.

![image](https://github.com/user-attachments/assets/df498e11-bd7f-4f8c-a7f9-b0a3b06c6fdb)


---

### Model Accuracy by Vectorization (Including Ensemble)

This visualization includes an **Ensemble model** combining predictions from multiple classifiers.

- Ensemble models outperform individual classifiers slightly.
- Shows that combining weaker models can lead to better predictive performance.

![image](https://github.com/user-attachments/assets/a83a5c44-40c6-49bc-8fd6-c899a6f3ade8)


---

## Transformer Model Performance

### Training and Validation Curves: ModernBERT vs DeBERTa

This plot displays:
- Training Loss vs Steps
- Validation Loss vs Steps
- Accuracy vs Steps

Comparison:

- **DeBERTa** outperforms **ModernBERT** consistently.
- DeBERTa shows lower training and validation loss, indicating better learning and generalization.
- DeBERTa achieves higher final accuracy (~82%) compared to ModernBERT (~80%).

Insights:

- Transformer-based models, when fine-tuned carefully, outperform traditional machine learning models.
- Validation curves are stable, suggesting minimal overfitting.

![image](https://github.com/user-attachments/assets/3036e760-b2fb-42cc-99ce-7ee74bbaa5ef)


---

## Critical Analysis

- Traditional ML models (SGD, Logistic Regression) perform reasonably well but plateau around 70% accuracy.
- Transformer models (DeBERTa) significantly outperform traditional ML, achieving higher accuracy and better generalization.
- Topic modeling highlights key differentiators between positive and negative reviews:
  - Positive reviews focus on product taste, quality, and experience.
  - Negative reviews highlight issues like manufacturing defects and customer service problems.

---

## Example Usage

After running the notebook:
- Analyze the review sentiments.
- Fine-tune models and visualize performances.
- Generate Kaggle submission files for predictions.

---

## Contributing

Pull requests are welcome.  
Feel free to open an issue to discuss new features or improvements.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgments

- Hugging Face for pre-trained models
- Scikit-learn for ML utilities
- Amazon open datasets for real-world reviews
- Gensim and NLTK for text analysis tools
