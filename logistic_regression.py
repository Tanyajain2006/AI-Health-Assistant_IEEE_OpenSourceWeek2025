import pandas as pd
import sys

try:
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import accuracy_score
except ImportError as e:
	raise ImportError(
		"Missing dependency: scikit-learn is not installed.\n"
		"Install it with `pip install scikit-learn` or `pip install -r requirements.txt`"
	) from e

try:
	import joblib
except ImportError:
	# joblib is typically installed as a scikit-learn dependency; provide guidance if missing
	print("Warning: `joblib` not found. Install it with `pip install joblib` if saving the model fails.")


df = pd.read_csv('intents.csv')

X = df["patterns"]
y = df["tag"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

preds = model.predict(X_vec)

print("Logistic Regression Accuracy:", accuracy_score(y, preds))

# Save both the trained model and the vectorizer for later use
try:
	joblib.dump(model, "chatbot_model.joblib")
	joblib.dump(vectorizer, "vectorizer.joblib")
	print("Model and vectorizer saved as 'chatbot_model.joblib' and 'vectorizer.joblib'")
except Exception as e:
	print("Failed to save model/vectorizer with joblib:", e, file=sys.stderr)
	print("You can install joblib or save the model another way.")