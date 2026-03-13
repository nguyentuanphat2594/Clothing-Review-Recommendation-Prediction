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

    st.title("📊 Tổng quan dữ liệu")

    # Load data
    data = pd.read_csv("Clothing-Review.csv")

    if data is not None:

        try:

            # Kiểm tra dữ liệu

            st.subheader("🔍 Kiểm tra chất lượng dữ liệu")

            col1, col2, col3 = st.columns(3)

            total_rows = data.shape[0]
            duplicate_rows = data.duplicated().sum()

            with col1:
                st.metric("Số hàng dữ liệu", total_rows)

            with col2:
                st.metric("Hàng trùng lặp", duplicate_rows)

            if duplicate_rows > 0:
                data = data.drop_duplicates()

            with col3:
                st.metric("Sau khi xử lý", data.shape[0])

            st.divider()

            # Thông tin dataset
            st.subheader("📁 Thông tin Dataset")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Số dòng", data.shape[0])

            with col2:
                st.metric("Số cột", data.shape[1])

            st.write("**Danh sách các cột:**")
            st.dataframe(
                pd.DataFrame({"Columns": data.columns}),
                use_container_width=True
            )

            st.divider()

            # Thống kê mô tả

            st.subheader("📈 Thống kê mô tả dữ liệu định lượng")

            desc = data[['Age', 'Rating', 'Positive Feedback Count']].describe()

            st.dataframe(
                desc.style.background_gradient(cmap="Blues"),
                use_container_width=True
            )

            # Lưu data dùng chung
            st.session_state['data'] = data

        except Exception as e:
            st.error("Có lỗi xảy ra khi xử lý dữ liệu")
            st.stop()

if page == "EDA":
    st.title("🔍 Exploratory Data Analysis (EDA)")

    data = st.session_state.data
    if data is None:
        st.stop()


    # Chọn sản phẩm

    class_list = ["Tất cả sản phẩm"] + list(data["Class Name"].dropna().unique())

    product = st.selectbox(
        "Chọn loại sản phẩm để phân tích:",
        class_list
    )

    if product == "Tất cả sản phẩm":
        data_plot = data
    else:
        data_plot = data[data["Class Name"] == product]


    # Biểu đồ rating + recommend

    st.subheader("📊 Tổng lượt đánh giá và khuyên dùng")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.countplot(data=data_plot, x='Rating', ax=axes[0])
    axes[0].set_title(f'Rating - {product}')
    axes[0].set_xlabel('Rating (sao)')
    axes[0].set_ylabel('Số lượt')

    sns.countplot(data=data_plot, x="Recommended IND", ax=axes[1])
    axes[1].set_title(f'Recommendation - {product}')
    axes[1].set_xlabel('Khuyên dùng (0 = Không, 1 = Có)')
    axes[1].set_ylabel('Số lượt')

    plt.tight_layout()
    st.pyplot(fig)


    # Plotly style

    pio.templates["transparent"] = pio.templates["simple_white"]
    pio.templates["transparent"].layout.paper_bgcolor = 'rgba(0,0,0,0)'
    pio.templates["transparent"].layout.plot_bgcolor  = 'rgba(0,0,0,0)'
    pio.templates.default = "transparent"


    # Age vs Recommendation

    st.subheader("📈 Khuyên dùng theo độ tuổi")

    fig = px.histogram(
        data_plot,
        x="Age",
        color="Recommended IND",
        marginal="box",
        nbins=65-18,
        color_discrete_sequence=['aqua', 'violet']
    )

    fig.update_layout(
        bargap=0.2,
        font=dict(color='black')
    )

    fig.update_traces(
        marker=dict(
            line=dict(color='black', width=1)
        )
    )

    st.plotly_chart(fig, use_container_width=True)


    # Age vs Rating

    st.subheader("📊 Rating theo độ tuổi")

    fig = px.histogram(
        data_plot,
        x="Age",
        color="Rating",
        marginal="box",
        nbins=65-18,
        color_discrete_sequence=[
            'aliceblue', 'skyblue', 'turquoise', 'aqua', 'darkcyan'
        ]
    )

    fig.update_layout(
        bargap=0.2,
        font=dict(color='black')
    )

    fig.update_traces(
        marker=dict(
            line=dict(color='black', width=1)
        )
    )

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
    