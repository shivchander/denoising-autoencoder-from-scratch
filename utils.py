__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
Utility functions for the homework
'''

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def parse_data(feature_file, label_file):
    """
    :param feature_file: Tab delimited feature vector file
    :param label_file: class label
    :return: dataset as a pandas dataframe (features+label)
    """
    features = pd.read_csv(feature_file, sep="\t", header=None)
    labels = pd.read_csv(label_file, header=None)
    features['label'] = labels
    return features


def split_data(dataset):
    """
    Randomly choose 4,000 data points from the data files to form a training set, and use the remaining
    1,000 data points to form a test set. Make sure each digit has equal number of points in each set
    (i.e., the training set should have 400 0s, 400 1s, 400 2s, etc., and the test set should have 100 0s,
    100 1s, 100 2s, etc.)
    :param dataset: pandas datafrome (features+label)
    :return: None. Saves Train and Test datasets as CSV
    """
    # init empty dfs
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for i in range(0, 10):
        df = dataset.loc[dataset['label'] == i]
        train_split = df.sample(frac=0.8, random_state=200)
        test_split = df.drop(train_split.index)
        train_df = pd.concat([train_df, train_split])
        test_df = pd.concat([test_df, test_split])

    train_df.to_csv('dataset/MNIST_Train.csv', sep=',', index=False)
    test_df.to_csv('dataset/MNIST_Test.csv', sep=',', index=False)


def get_train_test(train_file, test_file):
    train = pd.read_csv(train_file, sep=",")
    y_train = train['label'].values.reshape(4000, 1)
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)
    X_train = train.iloc[:, :-1].values

    test = pd.read_csv(test_file, sep=",")
    y_test = test['label'].values.reshape(1000, 1)
    y_test = mlb.fit_transform(y_test)
    X_test = test.iloc[:, :-1].values

    return X_train, y_train, X_test, y_test


def add_mask_noise(X_train, X_test, f0, f1):
    def noisify(data):
        noisy_data = np.copy(data)
        for row in noisy_data:
            zero_indices = np.random.choice(np.arange(784), int(f0*784), replace=False)
            one_indices = np.random.choice(list(set(np.arange(784)) - set(zero_indices)), int(f1*784), replace=False)
            row[zero_indices] = 0
            row[one_indices] = 1
        return noisy_data

    return noisify(X_train), noisify(X_test)


def train_test_digit_error(model, X_train, X_test):
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_cost = {}
    test_cost = {}
    for i, e in enumerate(range(0, 4000, 400)):
        train_cost[i] = model.compute_cost(train_preds[e:e + 400, ].T, X_train[e:e + 400, ].T)

    for i, e in enumerate(range(0, 1000, 100)):
        test_cost[i] = model.compute_cost(test_preds[e:e + 100, ].T, X_test[e:e + 100, ].T)

    return train_cost, test_cost


def plot_train_test_error(model, X_train, X_test):
    train_errors, test_errors = train_test_digit_error(model, X_train, X_test)
    X = np.arange(10)
    plt.bar(X + 0.00, list(train_errors.values()), width = 0.25, label='Train')
    plt.bar(X + 0.25, list(test_errors.values()), width = 0.25, label='Test')
    plt.xticks(X)
    plt.title('MSE Loss for each digit')
    plt.xlabel('Digits')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('figs/digitwise_train_test_error.pdf')
    plt.clf()


def random_outputs(model, noisy_test, clean_test):
    idx = np.random.randint(1000, size=8)
    sample_noisy_X = noisy_test[idx, :]
    sample_clean_X = clean_test[idx, :]
    sample_pred = model.predict(sample_noisy_X)

    fig, ax = plt.subplots(2, 8)
    for i in range(8):
        ax[0][i].imshow(sample_clean_X[i].reshape(28, 28).T, cmap='gray')
        ax[1][i].imshow(sample_pred[i].reshape(28, 28).T, cmap='gray')
    plt.savefig('figs/random_output.pdf')


def plot_features(model_weights, title):
    idx = np.random.randint(200, size=20)
    weights = model_weights[idx, :]

    fig, ax = plt.subplots(4, 5)
    pos = 0
    for i in range(4):
        for j in range(5):
            x = (weights[pos] - np.min(weights[pos]))/np.ptp(weights[pos])
            ax[i][j].imshow(x.reshape(28,28).T, cmap='gray')
            pos+=1
    fig.suptitle(title)
    fig.savefig(title+'.pdf')
    fig.clf()


def plot_confusion_matrix(y_true, y_pred, file_name):

    cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    df_cm = pd.DataFrame(cm, range(10), range(10))
    sns.heatmap(df_cm, annot=True, fmt='d')
    plt.show()
    plt.savefig(file_name+'.pdf')
    plt.clf()
