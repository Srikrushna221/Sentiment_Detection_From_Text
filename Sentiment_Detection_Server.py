from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.constants import DataTypes
from flask_ml.flask_ml_server.models import (FileInput, FileResult,
                                             ImageResult, ResponseModel,
                                             TextInput, TextResult)

from Sentiment_Detection_Models import (DistilbertSentimentModel,
                                        LRClassifierSentimentModel)

server = MLServer(__name__)

distilbert_sentiment_model = DistilbertSentimentModel()

lr_classifier_sentiment_model = LRClassifierSentimentModel()

##text_input = TextInput(text="Hello")
##print(distilbert_sentiment_model.predict([text_input]))

##file_input = [FileInput(file_path="emotion_data_file_input.csv")]
##print(type(file_input))
##distilbert_sentiment_model.predict_file_input(file_input)


@server.route("/sentiment_analysis_using_distilbert_text_input", DataTypes.TEXT)
def sentiment_analysis_using_distilbert_text_input(
    inputs: list, parameters: dict
) -> dict:
    results = distilbert_sentiment_model.predict_text_input(inputs)
    results = [TextResult(text=res["text"], result=res["sentiment"]) for res in results]
    response = ResponseModel(results=results)
    return response.get_response()


@server.route("/sentiment_analysis_using_distilbert_file_input", DataTypes.CUSTOM)
def sentiment_analysis_using_distilbert_file_input(inputs: list, parameters: dict):
    results = distilbert_sentiment_model.predict_file_input(inputs)
    results = [
        ImageResult(file_path=results[0]["file_path"], result=results[0]["result"])
    ]
    response = ResponseModel(results=results)
    return response.get_response()


@server.route("/sentiment_analysis_using_lr_classifier_text_input", DataTypes.TEXT)
def sentiment_analysis_text_input(inputs: list, parameters: dict) -> dict:
    results = lr_classifier_sentiment_model.predict_text_input(inputs)
    results = [TextResult(text=res["text"], result=res["sentiment"]) for res in results]
    response = ResponseModel(results=results)
    return response.get_response()


@server.route("/sentiment_analysis_using_lr_classifier_file_input", DataTypes.CUSTOM)
def sentiment_analysis_file_input(inputs: list, parameters: dict):
    results = lr_classifier_sentiment_model.predict_file_input(inputs)
    results = [
        ImageResult(file_path=results[0]["file_path"], result=results[0]["result"])
    ]
    response = ResponseModel(results=results)
    return response.get_response()


server.run()
