import joblib
import neattext.functions as nfx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

df = pd.read_csv("emotion_dataset_raw.csv")
print("Value Counts of Emotions:\n" + str(df["Emotion"].value_counts()) + "\n")
# Data Cleaning
df["Clean_Text"] = df["Text"].apply(nfx.remove_userhandles)
df["Clean_Text"] = df["Clean_Text"].apply(nfx.remove_stopwords)
print(df.head() + "\n")
Xfeatures = df["Clean_Text"]
ylabels = df["Emotion"]
x_train, x_test, y_train, y_test = train_test_split(
    Xfeatures, ylabels, test_size=0.3, random_state=1
)
pipe_lr = Pipeline(steps=[("cv", CountVectorizer()), ("lr", LogisticRegression())])
pipe_lr.fit(x_train, y_train)
print("Accuracy on Test Dataset: " + str(pipe_lr.score(x_test, y_test)) + "\n")
print("Sentiment Classes of the model are: " + str(pipe_lr.classes_) + "\n")

# Save the model
pipeline_file = open("emotion_classifier_lr_model.pkl", "wb")
joblib.dump(pipe_lr, pipeline_file)
pipeline_file.close()
