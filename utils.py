import nltk 
import json
import numpy as np

from sklearn.model_selection import train_test_split

def generate_tuples_from_file(training_file_path: str, num_samples=10) -> list:
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

    with open(training_file_path, 'r', encoding='utf-8') as f:
        count = 0
        for review in f:
            if count == num_samples:
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

def create_index(all_train_data_X: list) -> list:
    """
    Given the training data, create a list of all the words in the training data.
    Args:
        all_train_data_X: a list of all the training data in the format [[word1, word2, ...], ...]
    Returns:
        vocab: a list of all the unique words in the training data
    """
    # figure out what our vocab is and what words correspond to what indices
    #([[]], [])

    vocab = set()
    documents = all_train_data_X[0]
    for document in documents:
        for word in document:
            if word not in vocab:
                vocab.add(word)
    return list(vocab)

def featurize(vocab: list, data_to_be_featurized_X: list, binary: bool = False, verbose: bool = False) -> list:
    """
    Create vectorized BoW representations of the given data.
    Args:
        vocab: a list of words in the vocabulary
        data_to_be_featurized_X: a list of data to be featurized in the format [[word1, word2, ...], ...]
        binary: whether or not to use binary features
        verbose: boolean for whether or not to print out progress
    Returns:
        a list of sparse vector representations of the data in the format [[count1, count2, ...], ...]
    """
    # using a Counter is essential to having this not take forever
    vectorized_data = []

    documents = data_to_be_featurized_X[0]
    for sample in documents:
        vector = []
        sample_counter = Counter(sample)
        for word in vocab:
            if binary:
                if word in sample_counter:
                    vector.append(1)
                    if verbose:
                        print("Added 1 to current vector.")
                else:
                    vector.append(0)
                    if verbose:
                        print("Added 0 to vectorized data.")
            else:
                vector.append(sample_counter[word])
                if verbose:
                    print(f"{word} was added with value {sample_counter[word]}")

        vectorized_data.append(vector)
        if verbose:
            print(f"Adding current vector: {vector} to vectorized data.")

    if verbose:
        print("Vectorization completed.")

    return vectorized_data

def get_one_hot_encodings(labels: list) -> list:
    """
    Given a list of labels, return a list of one-hot encodings.
    Args:
        labels: a list of labels
    Returns:
        a list of one-hot encodings of the labels
    """
    one_hot_encodings = []
    for label in labels:
        one_hot_encoding = [0.0, 0.0, 0.0, 0.0, 0.0]
        one_hot_encoding[label - 1] = 1.0
        one_hot_encodings.append(np.array(one_hot_encoding))
    return np.array(one_hot_encodings, dtype='float32')

from sklearn.metrics import classification_report

def get_prfa(dev_y: list, preds: list, verbose=False) -> tuple:
    """
    Calculate precision, recall, f1, and accuracy for a given set of predictions and labels.
    Args:
        dev_y: list of labels
        preds: list of predictions
        verbose: whether to print the metrics
    Returns:
        tuple of precision, recall, f1, and accuracy
    """

    report = classification_report(dev_y, preds, output_dict=True)
    accuracy = report['accuracy']
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1_score = report['macro avg']['f1-score']

    if verbose:
        print(precision, recall, f1_score, accuracy)

    return (precision, recall, f1_score, accuracy)