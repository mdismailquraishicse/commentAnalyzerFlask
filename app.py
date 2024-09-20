from flask import Flask, render_template
import pandas as pd

path = 'prediction.csv'
df = pd.read_csv(path)

app = Flask(__name__)

@app.route('/')
def index():
    class_senti = ['a', 'b', 'c', 'd']
    class_spam = ['No', 'Yes']
    return render_template('index.html', comments = df['comment'].values,
                           spam = df['pred_spam'].values,
                           senti = df['pred_sentiment'].values,
                           class_spam = class_spam, class_senti = class_senti,
                           n = len(df['comment'].values))

@app.route('/analyze')
def analyze():
    text  = 'this is the comment'
    return render_template('result.html',
                           text= text)

if __name__=='__main__':
    Flask.run(debug=True, reload=True)