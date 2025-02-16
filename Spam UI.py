import streamlit as st
import joblib
from Preprocessor import preprocess
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from io import StringIO
from PIL import Image
import base64
import io

st.set_page_config(page_title="Spam Shield", layout="wide")
logo_path = "logo.png"  # Update with the correct path if needed
st.sidebar.image(logo_path, use_column_width=True)

def logo_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

icon_image = Image.open("logo.png")

st.markdown(
    f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <img src="data:image/png;base64,{logo_to_base64(icon_image)}" style="width: 68px; height: auto;"/>
            <h1 style='color: grey;font-size: 68px;'>Spam Shield</h1>
        </div>
        <p>Protect your inbox from spam emails!</p>
    </div>
    """,
    unsafe_allow_html=True
)

## Model Selection and Prediction
def check_spam(model_choice):
    msg = st.session_state.email_text.strip()
    if not msg:
        st.warning("⚠️ Please enter a message!")
        st.session_state.result = None
        return

    with st.spinner("Processing..."):
        try:
            processed_msg = preprocess(msg)
            message_vector = vectorizer.transform([processed_msg]).toarray()

            # Ensure models are properly loaded
            if model_choice == "📊 Logistic Regression":
                model = lr_model
            elif model_choice == "📈 Naive Bayes":
                model = nb_model
            elif model_choice == "🌳 Random Forest":
                model = rf_model
            else:
                st.error("⚠️ Invalid model selection!")
                return

            # Make prediction
            prediction = model.predict(message_vector)[0]
            confidence = model.predict_proba(message_vector)[0][prediction] * 100

            st.session_state.result = {
                "is_spam": prediction == 1,
                "confidence": confidence,
                "word_count": len(msg.split()),
                "char_count": len(msg)
            }

            # Show confidence with Pie Chart
            show_confidence_pie_chart(confidence)

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.session_state.result = None


def show_confidence_pie_chart(confidence):
    # Pie chart for confidence
    fig = go.Figure(data=[go.Pie(
        labels=["Not Spam", "Spam"],
        values=[confidence, 100 - confidence],
        marker_colors=["#4CAF50", "#ff4b4b"],
        hole=0.3
    )])

    fig.update_layout(
        title=f"Confidence: {confidence:.2f}%",
        showlegend=False,
        width=500,
        height=400
    )

    st.plotly_chart(fig)

# Loading models
@st.cache_resource
def load_models():
    nb_model = joblib.load('nb_model.joblib')
    lr_model = joblib.load('spam_model.joblib')
    rf_model = joblib.load('rf_model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    return lr_model, nb_model, rf_model, vectorizer

try:
    lr_model, nb_model, rf_model, vectorizer = load_models()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

#Sidebar
st.sidebar.header("Spam Shield")
sidebar_option = st.sidebar.radio(
    "📌 Select an Option:",
    ["🏠 Home", "📖 Instructions", "ℹ️ About"]
)

if sidebar_option == "🏠 Home":
    st.sidebar.subheader("⚙️ Options")

    # Model Selection Dropdown
    model_choice = st.sidebar.radio(
        "🧠 Select Model:",
        ["📊 Logistic Regression", "📈 Naive Bayes", "🌳 Random Forest"]
    )

    # Dark Mode
    dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)

    if dark_mode:
        st.markdown("""
            <style>
                body { background-color: #0e1117; color: white; }
                textarea { background-color: #1e1e1e; color: white; }
            </style>
        """, unsafe_allow_html=True)

    st.subheader("📩 Drop Your E-Mail Below:")

    if 'result' not in st.session_state:
        st.session_state.result = None

    # Input text area
    email_input = st.text_area(
        label="Enter your email text here:",
        value="",
        height=300,
        key="email_text"
    )

    # Check button
    if st.button("Check Spam!", type="primary", key="check_button"):
        check_spam(model_choice)

    # Result display section
    if st.session_state.result:
        result_data = st.session_state.result
        color = "#ff4b4b" if result_data["is_spam"] else "#4CAF50"
        result_text = "🚨 Spam" if result_data["is_spam"] else "✅ Not Spam"


        st.markdown(f"""
            <div style="border: 2px solid {color}; padding: 10px; border-radius: 10px; text-align: center; margin-top: 20px;">
                <h2 style="color: {color};">{result_text}</h2>
                <p><strong>Confidence:</strong> {result_data['confidence']:.2f}%</p>
                <p><strong>Word Count:</strong> {result_data['word_count']} | 
                <strong>Character Count:</strong> {result_data['char_count']}</p>
            </div>
        """, unsafe_allow_html=True)

        #Download CSV Result File
        if st.button("Download Results as CSV"):
            df = pd.DataFrame([result_data])
            csv = df.to_csv(index=False)
            st.download_button(label="Download as CSV", data=csv, file_name="spam_shield_results.csv", mime="text/csv")

elif sidebar_option == "📖 Instructions":
    st.markdown("""
        # 📖 How to Use the Spam Shield?

        ## Step-by-Step Instructions:

        1. **Enter Email Text**:
            - Type or paste the text of the email into the text area provided.
            - Ensure that the email text is clear and complete for accurate classification.

        2. **Select Classification Model**:
            - Choose between three models: **Logistic Regression**, **Naive Bayes**, or **Random Forest**.
            - The Logistic Regression model uses statistical relationships between words and spam likelihood, while the Naive Bayes model relies on the frequency of specific words in the email.
            - The **Random Forest** model is an ensemble learning technique that uses multiple decision trees to improve classification accuracy by considering multiple factors and averaging the results from different trees. This model is often more robust and accurate in various situations.

        3. **Click "Check Spam!"**:
            - After entering the email and selecting a model, click the **Check Spam!** button to start the classification process.
            - The app will process the email, classify it as **Spam** or **Not Spam**, and display the result.

        4. **View Results**:
            - The result will show the classification (Spam or Not Spam) along with the **confidence score** (in percentage), **word count**, and **character count** of the email.
            - A **confidence pie chart** will also appear, showing the proportion of confidence for both classes (Spam and Not Spam).

        5. **Download the Results**:
            - After classification, you can download the results as a **CSV file**. This file will contain details like whether the email is spam, confidence score, word count, and character count for later analysis or record-keeping.

        ## Additional Features:

        - **Dark Mode**: You can enable dark mode via the sidebar for a darker interface, providing a better experience in low-light environments.

        - **Model Switching**: You can switch between the Logistic Regression, Naive Bayes, and Random Forest models to compare their performances in classifying emails.

        - **Interactive Pie Chart**: The app displays a confidence pie chart showing the probability of the email being spam or not spam based on the selected model.

        ## Tips for Better Results:

        - **Clear Text**: Ensure the email content is correctly entered without typos or errors for better classification.
        - **Multiple Tests**: You can test different emails to evaluate the models' accuracy.
        - **Feedback**: If you notice patterns or false positives/negatives, consider using feedback to improve future iterations of the model (not implemented yet but planned).

        ## Troubleshooting:

        - **Empty Input**: Ensure that you enter some email text. If left empty, the app will ask you to input a message.
        - **Slow Processing**: If the app takes longer than expected to classify, it could be due to large email sizes or server load. Wait a moment, and the classification should complete soon.

    """)


elif sidebar_option == "ℹ️ About":
    st.markdown("""
        # ℹ️ About the Spam Shield

        **Spam Shield** is a machine learning application designed to classify whether an email is spam or not. This application leverages advanced Natural Language Processing (NLP) techniques and machine learning algorithms to make predictions based on the content of the email text.

        ## How It Works:
        The app uses three machine learning models to classify emails as spam or not:
        - **Logistic Regression** model, which has been trained on a large dataset of labeled emails (spam and non-spam). The model is designed to detect patterns and keywords in the email text that indicate whether the email is likely to be spam.
        - **Naive Bayes** model, which is another widely used classifier in text-based applications. It works on the principle of conditional probability and helps classify the email based on the features in the email content, such as the frequency of specific words.
        - **Random Forest** model, which is an ensemble learning technique that combines multiple decision trees to improve classification accuracy. The model uses the collective output of these decision trees to make a final prediction, often providing more reliable results compared to individual models. It is particularly effective in handling a large variety of features and noise in the data.

        All three models are designed to work in tandem, and the app allows you to switch between them to get a more reliable result based on different algorithms.

        The application uses a **text vectorizer** to process and convert the email content into a numerical format that the model can understand. Based on the processed input, the model predicts whether the email is spam or not.

        ## Main Features:
        - **Real-time Classification**: Simply paste or type the email text, and the app will classify it as spam or not.
        - **Confidence Score**: The app provides a confidence level for the prediction, helping you assess the reliability of the result.
        - **Word and Character Count**: The app also gives insights into the length of the email, including the word and character count.
        - **Model Switching**: Choose between Logistic Regression, Naive Bayes, and Random Forest for classification.
        - **Interactive Pie Chart**: View a confidence pie chart that visually represents the prediction's confidence level.
        - **Download Results**: Users can download the result in CSV format for further analysis.

        ## Extra Features:
        - **Dark Mode**: Enable dark mode for a more comfortable and modern user interface.
        - **Multiple Model Support**: Switch between multiple machine learning models (Logistic Regression, Naive Bayes, and Random Forest) to see how each performs.
        - **Session State**: The application uses session state to remember user inputs and results across multiple interactions.
        - **User-friendly Interface**: The application is designed with ease of use in mind, making it accessible for both novice and advanced users.

        ## Technology Stack:
        - **Backend**: Python (with libraries such as Scikit-learn, Numpy, and Pandas)
        - **Models**: Logistic Regression, Naive Bayes, and Random Forest classifiers for spam detection
        - **Text Processing**: Tfidf Vectorizer to convert email text into numerical features
        - **Framework**: Streamlit for creating an interactive web interface

        ## Machine Learning Models:
        - **Logistic Regression**: A linear model trained on a large dataset of labeled emails. It uses statistical relationships between words and spam likelihood.
        - **Naive Bayes**: A probabilistic classifier based on Bayes’ theorem. It considers the frequency of words in emails to classify them into spam or not spam.
        - **Random Forest**: An ensemble model that uses multiple decision trees to classify an email. It averages the predictions from several decision trees to improve accuracy and robustness, making it a great choice for complex classification tasks.

        All models were evaluated using standard classification metrics such as **accuracy** and **precision**. The models were then saved for easy integration into this web application.

        ## Created by:
        - **Wasif Sohail**
        - **Email**: wasifsohail66@gmail.com
        - **LinkedIn**: https://www.linkedin.com/in/wasif-sohail-4381602b4
        - **GitHub**: https://github.com/WasifSohail5

        ## Future Improvements:
        - **Model Improvement**: Explore advanced models like Neural Networks for better classification.
        - **Spam Detection Updates**: Regular updates to the dataset to improve spam detection accuracy.
        - **User Feedback**: Implement a feedback mechanism to improve the model's predictions over time.
    """)


