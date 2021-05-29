from flask import Flask, render_template, request
import pickle

tf = pickle.load(open('ressenttf.pkl','rb'))

classifier = pickle.load(open('resmnbclassifier.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def intro():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def index():
    review = request.form['review']
    data = [review]
    tfidf = tf.transform(data).toarray()
    prediction = classifier.predict(tfidf)
    return render_template('result.html', pred = prediction)

if __name__ == "__main__":
    app.run(debug=True)