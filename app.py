from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("text_df.csv")

# Feature extraction
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['text'])  # Fit the vectorizer to the training data
y = data['sentiment']

# Load pre-trained Random Forest classifier
with open('rf_sa.pkl', 'rb') as file:
    clf = pickle.load(file)

# Define routes
@app.route('/')
def index():
    # Calculate accuracy using the pre-trained model
    preds = clf.predict(X)
    accuracy = accuracy_score(y, preds)
    return render_template('index.html', accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    user_input = [request.form['text']]

    # Process user input for prediction
    vect_user_input = tfidf_vectorizer.transform(user_input)
    prediction = clf.predict(vect_user_input)

    # Render prediction template with the predicted sentiment
    return render_template('prediction.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)