# CineSentiment: BiLSTM Movie Review Classifier

## 🎬 IMDb Movie Review Sentiment Analysis using Bi-LSTM

This project is a deep learning-based sentiment analysis model designed to classify IMDb movie reviews as **positive** or **negative**. Leveraging a **Bidirectional Long Short-Term Memory (Bi-LSTM)** network, this model demonstrates significantly improved performance over traditional LSTM-based approaches.

---

## 📌 Project Highlights

- **🔍 Advanced NLP Pipeline:**  
  Implements tokenization, stop-word removal, lemmatization, and text normalization for effective preprocessing.

- **🧠 Deep Learning Architecture:**  
  Uses a Bi-LSTM model to capture both forward and backward dependencies in review texts.

- **📊 High Accuracy:**  
  Achieved **89.19% accuracy**, outperforming the baseline LSTM model referenced in prior research.

- **✅ Robust Evaluation:**  
  Applied **Stratified K-Fold Cross-Validation** for consistent and fair model evaluation.

---

## 🛠️ Technologies Used

- Python 3.x  
- TensorFlow / Keras  
- NLTK / spaCy  
- NumPy, Pandas  
- Scikit-learn  
- Matplotlib / Seaborn

---

## 📂 Dataset  
- **Source**: IMDb Movie Reviews  
- **Download Link**: [IMDb Dataset of 50K Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/)  
- **Labels**: Positive and Negative Sentiments  
- **Preprocessing**: HTML tag removal, tokenization, text encoding, and padding  

📌 *Since the dataset is too large to be uploaded to GitHub, you can download it from the link above before running the project.*

---

## 📊 Performance Metrics  

| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|----------|--------|---------|
| **Bi-LSTM (Ours)**  | **89.19%** | **89.21%** | **89.25%** | **89.23%** |
| LSTM (Reference)    | 87%       | 81%       | 80%    | 80%    |

---

## 🧪 How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/CineSentiment.git

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

#### Note: You may need to download NLTK resources like stopwords, punkt, etc.

## 📁 Project Structure
📂 IMDb-Sentiment-BiLSTM

├── 📄 sentiment_analysis_bilstm.py

├── 📄 preprocess.py

├── 📄 model_utils.py

├── 📄 requirements.txt

├── 📄 README.md

└── 📁 data/

  └── imdb_reviews.csv

## 📈 Results

- Accuracy: 89.19%

- Model Used: Bi-LSTM

- Evaluation Method: 5-fold Stratified Cross-Validation

A detailed comparison with baseline LSTM models is included in the results section of the notebook.

## 📌 Future Enhancements

- Incorporate attention mechanisms to improve interpretability

- Experiment with transformer-based models like BERT

- Deploy as a web app using Flask or Streamlit

