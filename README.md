
# 📰 Fake News Detection using Machine Learning & NLP

## 📌 Project Overview
With the rise of digital media, **misinformation and fake news** have become serious challenges.  
This project applies **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques to classify news articles as **Real** or **Fake**.  

We use **TF-IDF vectorization** with multiple ML models to detect fake news effectively.  
The best model achieved an accuracy of **~99%** on the dataset.  

---

## 📊 Dataset
- **True.csv** → Contains real news articles  
- **Fake.csv** → Contains fake news articles  
- Each file includes:
  - **title** → headline of the article  
  - **text** → full article text  
  - **label** → (1 = Real, 0 = Fake)  

**Source:** Kaggle Fake & True News Dataset  

---

## ⚙️ Workflow
1. **Data Preprocessing**
   - Combine title + text
   - Clean text (remove punctuation, URLs, stopwords, lowercase)
   - Tokenize and vectorize using **TF-IDF**

2. **Exploratory Data Analysis (EDA)**
   - Class distribution (real vs fake)
   - Text length analysis
   - Word frequency visualization

3. **Model Training**
   - Logistic Regression  
   - Decision Tree  
   - Random Forest  
   - Gradient Boosting  

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score  
   - Confusion Matrices  
   - Final model comparison bar chart  

---

## 🚀 Results
| Model                | Accuracy |
|-----------------------|----------|
| Logistic Regression   | ~0.98    |
| Decision Tree         | ~0.96    |
| Random Forest         | ~0.98    |
| Gradient Boosting     | ~0.99    |

- **Gradient Boosting** performed the best, achieving **~99% accuracy**.  
- Confusion matrices show very low misclassification.  

---

## 🛠️ Technologies Used
- **Python**  
- **Pandas, NumPy** – Data handling  
- **Matplotlib, Seaborn** – Visualization  
- **scikit-learn** – Machine Learning  
- **NLTK** – Text preprocessing  

---

## 📂 Project Structure
├── Fake.csv # Fake news data
├── True.csv # True news data
├── Fake_News_Detection.ipynb # Jupyter Notebook with full project
├── requirements.txt # Required libraries
└── README.md # Project documentation
