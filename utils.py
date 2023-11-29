import nltk 
import json

from sklearn.model_selection import train_test_split

def generate_tuples_from_file(training_file_path: str, testing=False) -> list:
    """
    Generates data from file formated like:

    tokenized text from file: [[word1, word2, ...], [word1, word2, ...], ...]
    labels: [0, 1, 0, 1, ...]
    
    Parameters:
        training_file_path - str path to file to read in
    Return:
        a list of lists of tokens and a list of int labels
    """
    X = []
    y = []

    with open(training_file_path, 'r') as f:
        count = 0
        for review in f:
            if count == 10 and testing:
                break
            review_as_dict = json.loads(review)

            stars = int(review_as_dict["stars"])
            text = review_as_dict["text"].replace("\n", " ")

            X.append(nltk.word_tokenize(text))
            y.append(stars)

            count += 1
    f.close()  
    return (X, y)

def split_data(data, test_size=0.2):
    """
    Splits data into training and test sets

    Parameters:
        data - tuple of lists of tokens and list of int labels
    Return:
        tuple of training and test sets
    """
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return ((X_train, y_train), (X_test, y_test))