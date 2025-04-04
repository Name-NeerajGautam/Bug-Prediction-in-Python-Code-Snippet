import streamlit as st
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def clean_code(text):
    text=re.sub(r'[^a-zA-Z0-9_\s]',' ',text)# remove special characters
    tokens=word_tokenize(text.lower())#tokenization and lowercasing
    tokens=[word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)
st.title("Bug Prediction and Prevention System")
st.write("Enter a Python code snippet to check if it contain a Bug.")
user_input=st.text_area("Enter code snippet")
button=st.button('Predict!')
if button:
    vectorizer=pickle.load(open('Tfidf_vectorizer.pkl','rb'))
    model=pickle.load(open('rf_model.pkl','rb'))
    cleaned_code=clean_code(user_input)
    input_Tfidf= vectorizer.transform([cleaned_code])
    prediction=model.predict(input_Tfidf)[0]
    if prediction:
        st.error("Bug Detected in the code")
    else:
        st.success("No Bugs found in the code.")
    
    





