from flask import Flask, render_template
import pandas as pd

path = 'prediction.csv'
df = pd.read_csv(path)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', df = df)

@app.route('/analyze')
def analyze():
    text  = 'this is the comment'
    return render_template('result.html',
                           text= text)

if __name__=='__main__':
    Flask.run(debug=True, reload=True)