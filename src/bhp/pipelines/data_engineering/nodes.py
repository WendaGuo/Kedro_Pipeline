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

PLEASE DELETE THIS FILE ONCE YOU START WORKING ON YOUR OWN PROJECT!
"""

from typing import Any, Dict, List

import pandas as pd


def reScaling(data: pd.DataFrame, columns_to_be_extracted: List[str]) -> pd.DataFrame:
    """
    extract target columns from raw data set, normalize features by (X-mean(X))/std(X), and shuffle the DataFrame.
    """
    if not isinstance(columns_to_be_extracted, list):
        raise ValueError("Error when extracting column names form yaml!")
    data = data[columns_to_be_extracted].sample(frac=1).reset_index(drop=True)
    normalize = lambda df: (df-df.mean())/df.std()
    data.iloc[:,:-1] = normalize(data.iloc[:,:-1])
    return data


def dataSplit(data: pd.DataFrame, train_test_split_ratio: float) -> Dict[str, Any]:
    """
    split DataFrame to training set and test set by a given ratio
    """
    n = data.shape[0]
    test_n = int(n * train_test_split_ratio)
    train_data = data.iloc[test_n:, :].reset_index(drop=True)
    test_data = data.iloc[:test_n, :].reset_index(drop=True)

    train_data_x = train_data.iloc[:, :-1]
    train_data_y = train_data.iloc[:, -1]
    test_data_x = test_data.iloc[:, :-1]
    test_data_y = test_data.iloc[:, -1]

    return dict(
        train_x=train_data_x,
        train_y=train_data_y,
        test_x=test_data_x,
        test_y=test_data_y
    )





