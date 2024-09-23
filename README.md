# Sentiment Detection From Text

### 1. **Project Overview**
This project implements two different sentiment analysis models using a DistilBERT model and a Logistic Regression (LR) classifier. The models can be used to analyze text or file inputs for sentiment prediction. A Flask-ML server is used to deploy the models, and a command-line interface (CLI) is also provided for easier usage.

### 2. **Dependencies**
The following libraries are required to set up and run the project:

- `transformers`: Provides the DistilBERT model for sentiment analysis.
- `pandas`: Used for file handling and data manipulation.
- `flask-ml`: A custom library used to deploy machine learning models through a Flask server.
- `joblib`: For loading the pre-trained Logistic Regression model.
- `argparse`: For building the command-line interface.
- `scikit-learn`: Used for Logistic Regression model creation.
- `torch`: Required by the DistilBERT model for PyTorch backend support.

To install all dependencies, run:

```bash
pip install transformers pandas flask-ml joblib scikit-learn torch
```

### 3. **Environment Setup**
1. **Install Python (3.8 or later)**: Make sure you have Python installed on your system. You can download the latest version of Python from the official [Python website](https://www.python.org/).

2. **Install dependencies**: Use `pip` to install the required dependencies as mentioned above.

3. **Prepare the Logistic Regression model**:
   - Ensure the file `emotion_classifier_lr_model.pkl` is present in the project folder by running the `LR_Classifier_Sentiment_Detection_Model.py` script, as it is used by the `LRClassifierSentimentModel` Class to make predictions.
   - We have used `emotion_dataset_raw.csv` to train the Logistic Regression Classifier and stored the trained model in `emotion_classifier_lr_model.pkl` which is being accessed by the `LRClassifierSentimentModel` in `Sentiment_Detection_Models.py`.

4. **Ensure files for Flask-ML integration**:
   The `Sentiment_Detection_Server.py` and `Sentiment_Detection_Client.py` files are used to set up and interact with the Flask-ML server. Place them in the correct project structure.

### 4. **How to Run the Models**
#### A. **Using the Command-Line Interface (CLI)**

This project supports running the models directly from the command line using the script `Sentiment_Detection_Models.py`.

1. **Sentiment Analysis using DistilBERT**

   **Command for Text Input**:
   ```bash
   python Sentiment_Detection_Models.py --model distilbert --input_type text --input "I am happy" "I am sad"
   ```

   **Command for File Input**:
   ```bash
   python Sentiment_Detection_Models.py --model distilbert --input_type file --input emotion_data_file_input.csv
   ```

2. **Sentiment Analysis using Logistic Regression**

   **Command for Text Input**:
   ```bash
   python Sentiment_Detection_Models.py --model lr --input_type text --input "This is great" "This is terrible"
   ```

   **Command for File Input**:
   ```bash
   python Sentiment_Detection_Models.py --model lr --input_type file --input emotion_data_file_input.csv
   ```

#### B. **Using Flask-ML Server**

1. **Starting the Server**:
   To deploy the models on a Flask-ML server, you will need to run the provided `Sentiment_Detection_Server.py` script. This script sets up the server that listens for requests from the client (which will be covered next).

   **Command to run the server**:
   ```bash
   python Sentiment_Detection_Server.py
   ```

   This will start the server and host the models for API-based interaction.

2. **Using the Client**:
   The `Sentiment_Detection_Client.py` file provides an easy way to interact with the server. You can send requests from the client to the server for sentiment analysis.

   **Command to run the client**:
   ```bash
   python Sentiment_Detection_Client.py
   ```

### 5. **Explanation of Flask-ML Server and Client**
The Flask-ML Server and Client provide a flexible framework to deploy and interact with machine learning models. This architecture allows users to set up an API service that other applications can interact with. The server is responsible for loading the models and handling API requests, while the client is responsible for sending requests and displaying the results.

#### **Server Overview**:
- The server is built using the `flask_ml` library and uses pre-built classes like `MLServer`, `TextInput`, and `FileInput` to handle text and file inputs. It manages the loading of models and ensures that requests are processed efficiently.

#### **Client Overview**:
- The client is designed to interact with the server, sending requests either for text or file-based inputs. It simplifies interaction by providing easy-to-use commands for querying the server.

### 6. **File Structure**
Ensure your project is organized as follows:

```
Sentiment_Detection_From_Text/
│
├── Sentiment_Detection_Models.py
├── Sentiment_Detection_Server.py
├── Sentiment_Detection_Client.py
├── LR_Classifier_Sentiment_Detection_Model.py
├── emotion_dataset_raw.csv
├── emotion_classifier_lr_model.pkl
├── emotion_data_file_input.csv
```

### 7. **Results**

- When using text input method the output is printed in the console itself as list of dictionaries which contain the text and sentiment for the list of given text inputs.

- When using the file input method with the given `emotion_data_file_input.csv`, the results of the sentiment analysis are saved to a new CSV file. Below is a brief explanation of what happens after running the models:

  1. **For the DistilBERT Model**:
    - After processing the file using the DistilBERT model (`bhadresh-savani/distilbert-base-uncased-emotion`), a new file named `sentiment_analysis_using_distilbert_result.csv` is generated which contains the text and sentiment for each line of text in the input file.

  2. **For the Logistic Regression (LR) Model**:
    - After processing the file with the Logistic Regression classifier, a new file named `sentiment_analysis_using_lr_classifier_result.csv` is generated which contains the text and sentiment for each line of text in the input file.

Both models save the results to new files for easy access and further analysis.

### 8. **Usage Examples**
You can use the following examples to test the functionality of your models:
- Text Input:
  ```bash
  python Sentiment_Detection_Models.py --model distilbert --input_type text --input "I love this product" "I hate the weather today"
  ```

- File Input:
  ```bash
  python Sentiment_Detection_Models.py --model lr --input_type file --input emotion_data_file_input.csv
  ```

### 9. **Conclusion**
This project provides a flexible way to perform sentiment analysis using state-of-the-art models like DistilBERT and traditional classifiers like Logistic Regression. The support for both CLI and Flask-ML server allows for easy integration into various applications.
