import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
data = pd.read_csv("text_df.csv")
# Feature extraction
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['text'])  # Fit the vectorizer to the training data
y = data['sentiment']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Now predict user input
user_input = [input("")]
vect_user_input = tfidf_vectorizer.transform(user_input)
prediction = clf.predict(vect_user_input)
print("Predicted sentiment:", prediction)

pickle.dump(clf, open('rf_sa.pkl', 'wb'))