import argparse

import joblib
import pandas as pd
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.constants import DataTypes
from flask_ml.flask_ml_server.models import (FileInput, FileResult,
                                             ImageResult, ResponseModel,
                                             TextInput, TextResult)
from transformers import pipeline


class DistilbertSentimentModel:
    def predict_file_input(self, data: list[FileInput]) -> list[dict]:
        df = pd.read_csv(data[0].file_path)
        text_data = df["Text"].fillna("")
        text_list = text_data.to_list()
        res_df = pd.DataFrame(self.helper(text_list))
        res_df.to_csv("sentiment_analysis_using_distilbert_result.csv")
        return [
            {
                "file_path": data[0].file_path,
                "result": "sentiment_analysis_using_distilbert_result.csv",
            }
        ]

    def predict_text_input(self, data: list[TextInput]) -> list[dict]:
        text_list = []
        for t in data:
            text_list.append(t.text)
        return self.helper(text_list)

    def helper(self, text_list):
        classifier = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            return_all_scores=True,
        )
        prediction = classifier(text_list)
        res = []
        for ind, i in enumerate(prediction):
            maxi = 0
            maxs = 0
            for j in range(len(i)):
                if i[j]["score"] > maxs:
                    maxs = i[j]["score"]
                    maxi = j
            res.append({"text": text_list[ind], "sentiment": i[maxi]["label"]})
        return res


class LRClassifierSentimentModel:
    def predict_file_input(self, data: list[FileInput]) -> list[dict]:
        df = pd.read_csv(data[0].file_path)
        text_data = df["Text"].fillna("")
        text_list = text_data.to_list()
        res_df = pd.DataFrame(self.helper(text_list))
        res_df.to_csv("sentiment_analysis_using_lr_classifier_result.csv")
        return [
            {
                "file_path": data[0].file_path,
                "result": "sentiment_analysis_using_lr_classifier_result.csv",
            }
        ]

    def predict_text_input(self, data: list[TextInput]) -> list[dict]:
        text_list = []
        for t in data:
            text_list.append(t.text)
        return self.helper(text_list)

    def helper(self, text_list):
        classifier = joblib.load("emotion_classifier_lr_model.pkl")
        prediction = list(classifier.predict(text_list))
        res = []
        for ind, i in enumerate(prediction):
            res.append({"text": text_list[ind], "sentiment": i})
        return res


# Add argument parsing for command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis using DistilBERT and LR Classifier"
    )
    parser.add_argument(
        "--model",
        choices=["distilbert", "lr"],
        required=True,
        help="Select the model to use",
    )
    parser.add_argument(
        "--input_type",
        choices=["text", "file"],
        required=True,
        help="Input type (text or file)",
    )

    # Use nargs="+" to accept a list of strings for text input
    parser.add_argument(
        "--input", nargs="+", required=True, help="Input text list or file path"
    )

    args = parser.parse_args()

    if args.model == "distilbert":
        model = DistilbertSentimentModel()
    elif args.model == "lr":
        model = LRClassifierSentimentModel()

    if args.input_type == "text":
        result = model.helper(args.input)  # args.input is now a list of strings
        for res in result:
            print(f"Text: {res['text']}, Sentiment: {res['sentiment']}")
    elif args.input_type == "file":
        output_file = model.predict_file_input(
            [FileInput(file_path=args.input[0])]
        )  # For file input, take the first element of the list
        print(f"Results saved to: {output_file}")
