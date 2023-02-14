import pickle
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


model = pickle.load(open("https://github.com/Darasimi199/Spam-Detection-Web-App/blob/main/spam/Spam_2/SVM_model.pkl", 'rb'))

def main():
    st.title('Message Detection Solution')

    #load in the dataset
    df = pd.read_csv('https://raw.githubusercontent.com/Darasimi199/Spam-Detection-Web-App/main/spam/Spam_2/clean_df.csv')

    # Input variable 'message'
    message = st.text_input('Message')
    message = [message]

    # Fit transform text into vectorizer
    tf_vec = TfidfVectorizer().fit(df['messages'])
    features = tf_vec.transform(message)
    features = features.toarray()

    # Prediction code
    if st.button('Predict'):
        make_pred = model.predict(features)
        if make_pred[0]==0:
            st.success('This message is Ham')
        else:
            st.success('This message is Spam')

if __name__ == '__main__':
    main()
