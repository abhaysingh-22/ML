import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# (model, parameter grid)
models = {
    "SVM": (
        make_pipeline(StandardScaler(), SVC()),
        {"svc__C": [0.1, 1, 10], "svc__kernel": ["linear", "rbf"], "svc__gamma": ["scale", 0.01]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10], "min_samples_split": [2, 5]},
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.5, 1.0]},
    ),
}


def evaluate(model):
    pred = model.predict(X_test)
    cv = cross_val_score(model, X_train, y_train, cv=5).mean()
    return cv, accuracy_score(y_test, pred), f1_score(y_test, pred)


rows = []
for name, (model, grid) in models.items():
    model.fit(X_train, y_train)
    cv, acc, f1 = evaluate(model)
    rows.append([name, "Before", cv, acc, f1, "default"])

    gs = GridSearchCV(model, grid, cv=5, n_jobs=-1).fit(X_train, y_train)
    cv, acc, f1 = evaluate(gs.best_estimator_)
    rows.append([name, "After", cv, acc, f1, gs.best_params_])

df = pd.DataFrame(rows, columns=["Model", "Stage", "CV Acc", "Test Acc", "Test F1", "Params"])
pd.set_option("display.width", 200, "display.max_colwidth", 80)
print(df.round(4).to_string(index=False))