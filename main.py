#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
Denoising and classification of MNIST Dataset using Autoencoder implemented from scratch
'''

from utils import *
from autoencoder import AutoencoderNN
from classifier import ClassifierNN


if __name__ == '__main__':
    # data = parse_data('dataset/MNISTnumImages5000_balanced.txt', 'dataset/MNISTnumLabels5000_balanced.txt')
    # split_data(data)
    # train = pd.read_csv('dataset/MNIST_Train.csv', sep=",")
    # test = pd.read_csv('dataset/MNIST_Test.csv', sep=",")

    X_train, train_labels, X_test, test_labels = get_train_test('dataset/MNIST_Train.csv', 'dataset/MNIST_Test.csv')
    X_noisy_train, X_noisy_test = add_mask_noise(X_train, X_test, f0=0.4, f1=0.05)

    # q1 Denoising Autoencoder

    # model = AutoencoderNN()
    # _ = model.fit(X_noisy_train, X_train, 784, 200, 1, 784, learning_rate=0.01, batch_size=32, num_epochs=400, plot_error=True)
    # random_outputs(model, X_noisy_test, X_test)
    # plot_train_test_error(model, X_train, X_test)
    # train_test_digit_error(model, X_train, X_test)
    # plot_features(model.parameters['W1'], title='Denoising Autoencoder')

    # q2 Classifier
    # Case 1: Pretrained weights from HW3 Problem 2
    # model1 = AutoencoderNN()
    # model1.fit(X_train, X_train, 784, 200, 1, 784, learning_rate=0.01, batch_size=32, num_epochs=20, plot_error=True)
    #
    # model2 = ClassifierNN()
    # model2.fit(X_train, X_train, 784, 200, 1, 784, pre_trained_weights=model1.parameters,
    #            learning_rate=0.01, batch_size=32, num_epochs=150, plot_error=True)
    #
    # model3 = AutoencoderNN()
    # _ = model3.fit(X_noisy_train, X_train, 784, 200, 1, 784, learning_rate=0.01, batch_size=32, num_epochs=250,
    #                plot_error=True)
    #
    # model4 = ClassifierNN()
    # _ = model4.fit(X_train, train_labels, 784, 200, 1, 10, pre_trained_weights=model3.parameters,
    #                learning_rate=0.01, batch_size=32, num_epochs=250, plot_error=True)
    #
    # y1_train_preds = model2.predict(X_train)
    # y1_test_preds = model2.predict(X_test)
    #
    # y2_train_preds = model4.predict(X_train)
    # y2_test_preds = model4.predict(X_test)
    #
    # plot_confusion_matrix(train_labels, y1_train_preds, 'case1train_cm')
    # plot_confusion_matrix(test_labels, y1_test_preds, 'case1test_cm')
    # plot_confusion_matrix(train_labels, y2_train_preds, 'case2train_cm')
    # plot_confusion_matrix(test_labels, y2_test_preds, 'case2test_cm')

