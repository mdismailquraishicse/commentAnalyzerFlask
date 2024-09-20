from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import pickle
import numpy as np

sentimentAnalyzer = pickle.load(open('pickles/sentimentAnalyzer.pkl','rb'))
spamDetector = pickle.load(open('pickles/spam_model_with_pipe.pkl','rb'))

path = 'prediction.csv'
df = pd.read_csv(path)

app = Flask(__name__)
class_senti = ['Irrelevant', 'Natural', 'Negative', 'Positive']
class_spam = ['No Spam', 'Spam']

@app.route('/')
def index():
    return render_template('index.html', comments = df['comment'].values,
                           spam = df['pred_spam'].values,
                           senti = df['pred_sentiment'].values,
                           class_spam = class_spam, class_senti = class_senti,
                           n = len(df['comment'].values))

@app.route('/pred',methods=['POST', "GET"])
def pred():
    text = request.form.get('comment')
    spam = spamDetector.predict(np.array([text])) 
    spam = class_spam[spam[0]]

    senti = sentimentAnalyzer.predict(np.array([text])) 
    senti = class_senti[senti[0]]
    return render_template('prediction.html',
                           spam = spam, text = text, senti=senti)
@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__=='__main__':
    Flask.run(debug=True, reload=True)