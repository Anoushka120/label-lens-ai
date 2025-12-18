import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = {
    "text": [
        "whole wheat flour oats honey",
        "hydrogenated oil sugar syrup",
        "chickpeas olive oil spices",
        "corn syrup artificial flavor"
    ],
    "label": [1, 0, 1, 0]
}

df = pd.DataFrame(data)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

pipe.fit(df["text"], df["label"])
joblib.dump(pipe, "model.joblib")
print("Model saved")
