# 👗 Clothing Recommendation Prediction App (LSTM + Streamlit)

## 📌 Overview
This repository contains a **Streamlit web application** that predicts whether a customer would **recommend a clothing product** based on review comments using a pre-trained **LSTM Neural Network model** built with **TensorFlow/Keras**.

The app also provides **data quality statistics** and **exploratory visual analysis (EDA)** for business insights.

## 🎯 Project Goals
- Predict recommendation intent from customer review text (**RCM / Not RCM**)
- Deploy deep learning model into an **interactive real-time web demo**
- Perform **EDA and customer segmentation analysis**
- Validate structured input using a saved tokenizer and trained model

## ✨ Key Features
### 1. Data Statistics & Quality Checks
- Load review dataset (CSV)
- Detect duplicate records and remove if needed
- Display dataset dimensions and numeric summary statistics

### 2. Exploratory Data Analysis (EDA)
- Star rating distribution by product category
- Recommendation ratio visualization
- Customer insights via age-based segmentation
- Interactive charts using **Plotly** and static charts using **Matplotlib**

### 3. Recommendation Prediction
- Text input for single review inference
- Tokenized using a saved **Keras Tokenizer**
- Prediction returned as:
  - **Class label** (`RCM / Not RCM / Uncertain`)
  - **Probability score**

## 🧠 Model Details
| Component | Description |
|---|---|
| Model Type | LSTM Text Classification |
| Framework | TensorFlow / Keras |
| Input | Tokenized customer review text |
| Output | 1 sigmoid neuron (Recommendation probability) |
| Tokenizer | Saved Keras `Tokenizer` (`tokenizer.pkl`) |
| Training Method | Train / Validation / Test split |
| Text Processing | Lowercase, stopword removal, tokenization, lemmatization |
| Deployment Mode | Inference only (Model loaded without compile) |

## 🛠 Tech Stack
- **Frontend**: Streamlit  
- **Deep Learning**: TensorFlow, Keras (LSTM)  
- **Data Processing**: Pandas, NumPy  
- **Visualization**: Matplotlib, Plotly, Plotly Express  
- **NLP Tools**: NLTK (stopwords, tokenizer, lemmatizer)  
- **Evaluation**: Confusion Matrix, Classification Report (Scikit-Learn)  
- **Model Storage**: `.keras` model + `tokenizer.pkl`  
