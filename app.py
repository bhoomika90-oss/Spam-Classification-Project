from flask import Flask, render_template, request, redirect, session
import sqlite3
import pickle

app = Flask(__name__)
app.secret_key = "secret"

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# DB setup
conn = sqlite3.connect('database.db')
conn.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")
conn.close()

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form['username']
        pwd = request.form['password']

        if user == "admin" and pwd == "123":
            session['user'] = user
            return redirect('/dashboard')

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    msg = request.form['message']
    data = vectorizer.transform([msg])
    pred = model.predict(data)[0]

    if pred == 1:
        return render_template('spam.html', msg=msg)
    else:
        return render_template('notspam.html', msg=msg)

# Auto test feature
@app.route('/auto_test')
def auto_test():
    test = "Win free money now"
    data = vectorizer.transform([test])
    pred = model.predict(data)[0]

    result = "Spam" if pred == 1 else "Not Spam"
    return f"Auto Test Result: {result}"

if __name__ == "__main__":
    app.run(debug=True)