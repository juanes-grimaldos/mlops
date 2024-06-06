from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from typing import Callable, Dict, Tuple, Union
from scipy.sparse._csr import csr_matrix
from pandas import Series
import numpy as np
if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(training_set: Dict[str, Union[Series, csr_matrix]], *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    X_train, y_train, dv = training_set['build']

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)

    return lr, lr.intercept_

