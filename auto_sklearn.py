import numpy as np
from pprint import pprint

import sklearn.datasets
import sklearn.metrics
from sklearn.utils.multiclass import type_of_target

import autosklearn.classification
# Using reuters multilabel dataset -- https://www.openml.org/d/40594
X, y = sklearn.datasets.fetch_openml(data_id=40594, return_X_y=True, as_frame=False)

# fetch openml downloads a numpy array with TRUE/FALSE strings. Re-map it to
# integer dtype with ones and zeros
# This is to comply with Scikit-learn requirement:
# "Positive classes are indicated with 1 and negative classes with 0 or -1."
# More information on: https://scikit-learn.org/stable/modules/multiclass.html
y[y == "TRUE"] = 1
y[y == "FALSE"] = 0
y = y.astype(int)

# Using type of target is a good way to make sure your data
# is properly formatted
print(f"type_of_target={type_of_target(y)}")

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1
)

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,
    per_run_time_limit=30,
    # Bellow two flags are provided to speed up calculations
    # Not recommended for a real implementation
    initial_configurations_via_metalearning=0,
    smac_scenario_args={"runcount_limit": 1},
)
automl.fit(X_train, y_train, dataset_name="reuters")

print(automl.leaderboard())