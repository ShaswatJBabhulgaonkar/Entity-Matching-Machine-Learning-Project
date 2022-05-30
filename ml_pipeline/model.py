from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ("rand_forest", RandomForestClassifier(
        random_state=0
    )),
])
