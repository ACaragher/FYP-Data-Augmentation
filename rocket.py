from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline
from sktime.transformations.panel.rocket import Rocket, MiniRocketMultivariate

from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
import time
import numpy as np

def run_classifier(train_x, train_y, test_x, test_y):
    
    model = Pipeline(
            [
            ('rocket', Rocket(random_state=0,normalise=False)),
            ('normalise', StandardScaler()),
            ('model', RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
            ],
            verbose=True,
    )
    model.fit(train_x, train_y)
    preds = model.predict(test_x)
    acc = accuracy_score(preds, test_y) * 100
    
    return acc, preds