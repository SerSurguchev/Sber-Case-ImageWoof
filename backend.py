from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def about():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return 'Welcome to prediction'

if __name__ == '__main__':
    app.run(debug=True)