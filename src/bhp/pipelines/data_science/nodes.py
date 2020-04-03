# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""
# pylint: disable=invalid-name

import logging
from typing import Any, Dict
from datetime import datetime
import os
import numpy as np
import pandas as pd


def train_model(
    train_x: pd.DataFrame, train_y: pd.DataFrame, parameters: Dict[str, Any]
) -> pd.DataFrame:
    """Node for training a simple multi-class logistic regression model. The
    number of training iterations as well as the learning rate are taken from
    conf/project/parameters.yml. All of the data as well as the parameters
    will be provided to this function at the time of execution.
    """
    num_iter = parameters["example_num_train_iter"]
    eta = parameters["example_learning_rate"]
    X = train_x.to_numpy()
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    Y = train_y.to_numpy()
    # gradient descent
    theta = np.zeros(X.shape[1])
    MSE = []
    for _ in range(num_iter):
        theta -= eta * gradient(X, Y, theta)
        MSE.append(mse(X, Y, theta))

    now = datetime.now()
    now_time = datetime.strftime(now, '%Y-%m-%d-%H:%M:%S')
    cwd = os.getcwd()
    ind = cwd.find("src")

    # plt.plot(MSE, np.arange(1, len(MSE) + 1, 1))
    # plt.title("The training process")
    # plt.xlabel("training step")
    # plt.ylabel("Mean Squared Error")
    # figure_name = (cwd[:cwd.find("src")+1] + "figures/training curve saved at {}.png").format(now_time)
    # plt.savefig(figure_name)

    log = logging.getLogger(__name__)
    log.info("Model training complete, factor loading coefficients = {}".format(theta))

    columns = parameters["columns_to_be_extracted"][:-1]
    columns = ['bias']+columns
    theta = pd.DataFrame(theta[np.newaxis, :], columns=columns)
    theta.set_index(columns[0])
    return theta


def report_model(test_x: pd.DataFrame,
                 test_y: pd.DataFrame,
                 theta: pd.DataFrame) -> None:
    theta = theta.to_numpy()[0]
    X = test_x.to_numpy()
    bias = np.zeros((X.shape[0],1))
    X = np.concatenate((bias,X),axis=1)
    Y = test_y.to_numpy()
    MSE = mse(X,Y,theta)
    log = logging.getLogger(__name__)
    log.info("Model MSE on test set = \n{:.4f}".format(MSE))


def predict(model: np.ndarray, test_x: pd.DataFrame) -> np.ndarray:
    """Node for making predictions given a pre-trained model and a test set.
    """
    X = test_x.to_numpy()

    # Add bias to the features
    bias = np.ones((X.shape[0], 1))
    X = np.concatenate((bias, X), axis=1)

    # Predict "probabilities" for each class
    result = _sigmoid(np.dot(X, model))

    # Return the index of the class with max probability for all samples
    return np.argmax(result, axis=1)


def report_accuracy(predictions: np.ndarray, test_y: pd.DataFrame) -> None:
    """Node for reporting the accuracy of the predictions performed by the
    previous node. Notice that this function has no outputs, except logging.
    """
    # Get true class index
    target = np.argmax(test_y.to_numpy(), axis=1)
    # Calculate accuracy of predictions
    accuracy = np.sum(predictions == target) / target.shape[0]
    # Log the accuracy of the model
    log = logging.getLogger(__name__)
    log.info("Model accuracy on test set: %0.2f%%", accuracy * 100)


def _sigmoid(z):
    """A helper sigmoid function used by the training and the scoring nodes."""
    return 1 / (1 + np.exp(-z))


def mse(X: np.ndarray, Y: np.ndarray, theta: np.ndarray) -> float:
    return 1/2*np.sum(np.power(np.sum(theta[np.newaxis,:]*X,axis=1)-Y,2))/X.shape[0]


def gradient(X: np.ndarray, Y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    if (X.shape[0]!=Y.shape[0]):
        raise ValueError("Conflicting dimensions with X: {} but Y:{}".format(X.shape,Y.shape))
    kernel = np.sum(theta[np.newaxis,:]*X,axis=1)-Y
    return np.sum(kernel[:,np.newaxis]*X,axis=0)/X.shape[0]






