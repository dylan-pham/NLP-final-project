import nltk 
import json

def generate_tuples_from_file(training_file_path: str) -> list:
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
        for review in f:
            review_as_dict = json.loads(review)

            stars = int(review_as_dict["stars"])
            text = review_as_dict["text"].replace("\n", " ")

            X.append(nltk.word_tokenize(text))
            y.append(stars)
    f.close()  
    return (X, y)