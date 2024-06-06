from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """

    # Convert the dataframe into a list of dictionaries
    df_filtered = data.copy()
    train_dicts = df_filtered[['PULocationID', 'DOLocationID']].astype(str).to_dict(orient='records')

    # Fit a dictionary vectorizer
    dv = DictVectorizer()
    dv.fit(train_dicts)

    # Get the feature matrix
    X_train = dv.transform(train_dicts)

    # Get the y 
    y_train = df_filtered.duration.values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)

    intercept = lr.intercept_

    return intercept

