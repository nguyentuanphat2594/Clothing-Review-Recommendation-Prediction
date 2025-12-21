import warnings
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pickle
from tensorflow.keras.models import load_model

from ultils import predict_review

lemm = WordNetLemmatizer()

warnings.filterwarnings("ignore")
import plotly.io as pio

import streamlit as st

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1220;
        color: #e5e7eb;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.selectbox(
    "Chọn chức năng",
    ["Thống kê chung về dữ liệu mẫu", "EDA", "Dự đoán"]
)
if page == "Thống kê chung về dữ liệu mẫu":
    st.title("📂 Load data và Thống kê chung")
    # Load data
    data = data = pd.read_csv("Clothing-Review.csv")
    if data is not None:
        try:
            st.header("🔍 Kiểm tra chất lượng dữ liệu")
            st.write(f"**Số hàng dữ liệu:**")
            st.write(f"{data.shape[0]}")
            st.write(f'**Số hàng trùng lắp:**')
            st.write(f'{data.duplicated().sum()} hàng')
            if data.duplicated().sum() > 0:
                st.write("**Loại bỏ hàng trùng lắp:**")
                data = data.drop_duplicates()
                st.write(f'{data.shape[0]} hàng')

            st.header('Thông tin chung')
            st.write(f'**Số dòng dữ liệu:**')
            st.write(f'{data.shape[0]}')
            st.write(f'**Số cột dữ liệu:**')
            st.write(f'{data.shape[1]}')
            st.write("**Bao gồm:**")
            for col in data.columns:
                st.write(f'*{col}*')
            st.header("**Thống kê mô tả các dữ liệu định lượng:**")
            st.write(data[['Age', 'Rating', 'Positive Feedback Count']].describe()) # Thống kê mô tả

            # Lưu data dùng chung
            st.session_state['data'] = data
        except Exception as e:
            st.stop()

if page == "EDA":
    st.title("🔍 EDA")
    data = st.session_state.data
    if data is None:
        st.stop()
    st.write("**Nhập sản phẩm muốn thống kê theo số thứ tự:**")
    st.write("0. Tất cả sản phẩm")
    for i, value in enumerate(data['Class Name'].value_counts().index):
        st.write(f"{i+1}. {value}")

    choice = st.number_input(
        "Nhập số nguyên:",
        min_value=0,
        max_value=20,
        step=1
    )
    if choice == 0:
        pass
    else:
        product = data['Class Name'].value_counts().index[choice-1]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if 0 < choice < len(data['Class Name'].value_counts()):
        data_plot = data[data['Class Name'] == product]
    elif choice == 0:
        data_plot = data
        product = 'tất cả sản phẩm'

    st.subheader("Biểu đồ cột thể hiện tổng lượt đánh giá/khuyên dùng sản phẩm")

    # Biểu đồ cột thể hiện tổng lượt đánh giá sản phẩm
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(data = data_plot, x='Rating', ax=axes[0])
    axes[0].set_title(f'Tổng lượt đánh giá {product}')
    axes[0].set_xlabel('Rating (sao)')
    axes[0].set_ylabel('Số lượt đánh giá')

    
    # Biểu đồ cột thể hiện tổng lượt khuyên dùng sản phẩm
    sns.countplot(data = data_plot, x="Recommended IND", ax=axes[1])
    axes[1].set_title(f'Tổng lượt khuyên dùng {product}')
    axes[1].set_xlabel('Khuyên dùng (0 = Không, 1 = Có)')
    axes[1].set_ylabel('Số lượt')

    # Hiển thị
    plt.tight_layout()
    st.pyplot(fig)

    pio.templates["transparent"] = pio.templates["simple_white"]

    pio.templates["transparent"].layout.paper_bgcolor = 'rgba(0,0,0,0)'
    pio.templates["transparent"].layout.plot_bgcolor  = 'rgba(0,0,0,0)'

    pio.templates.default = "transparent"
    
    st.subheader("Biểu đồ thống kê lượng khuyên dùng theo độ tuổi")
    st.write("Khuyên dùng: 1")
    st.write("Không khuyên dùng: 0")
    fig = px.histogram(data, marginal='box',
                    x="Age",
                    color="Recommended IND",
                    nbins=65-18,
                    color_discrete_sequence=['aqua', 'violet'])
    fig.update_layout(
            bargap=0.2,
            font=dict(
            color='black'
        ))
    fig.update_traces(marker=dict(
            line=dict(
                color='black',
                width=1
            )))
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("Biểu đồ thống kê lượt đánh giá theo độ tuổi")
    fig = px.histogram(data,
                    x="Age",
                    marginal='box',
                    color="Rating",
                    nbins=65-18,
                    color_discrete_sequence
                    =['aliceblue', 'skyblue', 'turquoise', 'aqua', 'darkcyan'])
    fig.update_layout(
            font=dict(
                    color='black'
            ),
            bargap=0.2
        )
    fig.update_traces(
            marker=dict(
            line=dict(
                color='black',
                width=1
                )))
    st.plotly_chart(fig, use_container_width=True)


if page == "Dự đoán":
    st.title("🔮 Dự đoán khuyên dùng sản phẩm bằng mô hình LSTM")
    data = st.session_state.data
    if data is None:
        st.stop()
    model = load_model('model_LSTM.keras')
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    st.header("👗 Dự đoán khuyên dùng sản phẩm qua comment")
    st.write("**Nhập đánh giá (tiếng Anh)**")
    text = st.text_area("")

    if st.button("Dự đoán"):
        if text.strip() == "":
            st.warning("Vui lòng nhập đánh giá")
        else:
            label, prob = predict_review(text, model, tokenizer)
            st.success(label)
            st.write(f"Probability: **{prob:.2f}**")
    