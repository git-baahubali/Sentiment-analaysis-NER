import streamlit as st
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# * RUn the code in terminal < python -m nltk.downloader vader_lexicon >
# or
# * run the following code in python interpretor itself < nltk.download('vader_lexicon') >


import nltk
nltk.download('vader_lexicon')



def feedback(text): 
    sent = SentimentIntensityAnalyzer()
    score = sent.polarity_scores(text)['compound']
    
    '''feedback fn takes a string and returns if its positive ,neutral  or negative '''
    
    if score >= 0.1:
        return ("positive")
    elif score <= (-0.1):
        return ('Negative')
    else:
        return ("Neutral")
    

st.title("Sentiment analysis")

with st.form(key = "Sentiment_form"):
    sentiment_input =   st.text_input("Enter the sentence:", value='Five stars! Exceeded my expectations.')
    # st.write(sentiment_input)
    Submit_btn = st.form_submit_button(label='Submit ')
    if Submit_btn:
        st.write(feedback(sentiment_input))


# NER
import spacy
from spacy import displacy
st.title("Name Entity Recognition")

with st.form(key = "NER_form",):
    ner_input =   st.text_area("Enter the sentence:", value='''On June 22, 2024, Elon Musk, the CEO of SpaceX and Tesla, announced at a press conference in San Francisco, California, that SpaceX would be launching a new mission to Mars in collaboration with NASA and the European Space Agency (ESA). The mission, called 'Mars Odyssey,' is set to depart from Kennedy Space Center in Florida in December 2025. The project, funded by a $5 billion investment from the U.S. government and private investors like Jeff Bezos's Blue Origin, aims to establish the first human colony on Mars, advancing interplanetary exploration.''')
    
    
    Submit_btn = st.form_submit_button("Submit")
    
    if Submit_btn:
        # The code snippet `nltk.download('en_core_web_sm')` is attempting to download the English
        # language model for spaCy called `en_core_web_sm`. This model is used for various natural
        # language processing tasks, including Named Entity Recognition (NER).
        nltk.download('en_core_web_sm')
        ner = spacy.load('en_core_web_sm')
        entity = ner(ner_input)
        html_string = displacy.render(entity, style='ent')

        st.markdown(html_string, unsafe_allow_html = True)
