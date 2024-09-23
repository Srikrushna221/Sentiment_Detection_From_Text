import pandas as pd
from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.constants import DataTypes

client = MLClient("")  # Create an instance of the MLClient object


##data = pd.read_csv("emotion_dataset_raw.csv")
##
##s = data["Text"].head().fillna('')
##s = s.to_list()
##
##inputs = []
##
##for i in s:
##    inputs.append({"text":i})

##print(inputs[0:5])


inputs = [
    {"text": "Hello, I am happy"},
    {"text": "Feeling angry on that person"},
    {"text": "This is sad"},
    {"text": "I feel neutral"},
]  # The inputs to be sent to the server


SENTIMENT_MODEL_URL = (
    "http://127.0.0.1:5000/sentiment_analysis_using_distilbert_text_input"
)
data_type = DataTypes.TEXT  # The type of the input data
client.set_url(SENTIMENT_MODEL_URL)
response = client.request(inputs, data_type)
print("sentiment_analysis_using_distilbert_text_input response:")
print(response)


SENTIMENT_MODEL_URL = (
    "http://127.0.0.1:5000/sentiment_analysis_using_lr_classifier_text_input"
)
data_type = DataTypes.TEXT  # The type of the input data
client.set_url(SENTIMENT_MODEL_URL)
response = client.request(inputs, data_type)
print("sentiment_analysis_using_lr_classifier_text_input response:")
print(response)


SENTIMENT_MODEL_URL = (
    "http://127.0.0.1:5000/sentiment_analysis_using_distilbert_file_input"
)
client.set_url(SENTIMENT_MODEL_URL)
data_type = DataTypes.CUSTOM
inputs = [{"file_path": "emotion_data_file_input.csv"}]
response = client.request(inputs, data_type)
print("sentiment_analysis_using_distilbert_file_input response:")
print(response)

SENTIMENT_MODEL_URL = (
    "http://127.0.0.1:5000/sentiment_analysis_using_lr_classifier_file_input"
)
client.set_url(SENTIMENT_MODEL_URL)
data_type = DataTypes.CUSTOM
inputs = [{"file_path": "emotion_data_file_input.csv"}]
response = client.request(inputs, data_type)
print("sentiment_analysis_using_lr_classifier_file_input response:")
print(response)
