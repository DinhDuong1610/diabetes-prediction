import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# read data
data = pd.read_csv("../dataset/diabetes.csv")

# data split
target = "Outcome"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# performance testing with multiple parameter sets
params = [
    {
        "kernel": ["linear"],
        "C": [0.1, 1, 10, 100],
        "shrinking": [True, False],
        "probability": [True, False],
        "cache_size": [100, 200],
        "decision_function_shape": ["ovr"],
        "break_ties": [True, False],
    },
    {
        "kernel": ["linear"],
        "C": [0.1, 1, 10, 100],
        "shrinking": [True, False],
        "probability": [True, False],
        "cache_size": [100, 200],
        "decision_function_shape": ["ovo"],
        "break_ties": [False],  # Chỉ False hợp lệ với 'ovo'
    },

    # Kernel: poly (có degree, cần gamma)
    {
        "kernel": ["poly"],
        "C": [0.1, 1, 10],
        "degree": [2, 3, 4],
        "gamma": ["scale", "auto"],
        "shrinking": [True, False],
        "probability": [True, False],
        "cache_size": [100, 200],
        "decision_function_shape": ["ovr"],
        "break_ties": [True, False],
    },
    {
        "kernel": ["poly"],
        "C": [0.1, 1, 10],
        "degree": [2, 3, 4],
        "gamma": ["scale", "auto"],
        "shrinking": [True, False],
        "probability": [True, False],
        "cache_size": [100, 200],
        "decision_function_shape": ["ovo"],
        "break_ties": [False],
    },

    # Kernel: rbf & sigmoid (có gamma, không có degree)
    {
        "kernel": ["rbf", "sigmoid"],
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
        "shrinking": [True, False],
        "probability": [True, False],
        "cache_size": [100, 200],
        "decision_function_shape": ["ovr"],
        "break_ties": [True, False],
    },
    {
        "kernel": ["rbf", "sigmoid"],
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
        "shrinking": [True, False],
        "probability": [True, False],
        "cache_size": [100, 200],
        "decision_function_shape": ["ovo"],
        "break_ties": [False],
    },
]

grid_search = GridSearchCV(estimator=SVC(), param_grid=params, cv=5, scoring="recall", verbose=2)

grid_search.fit(x_train, y_train)
print(grid_search.best_estimator_)
print(grid_search.best_params_)
print(grid_search.best_score_)

# SVC(C=10, break_ties=True, cache_size=100, gamma='auto', probability=True)
# {'C': 10, 'break_ties': True, 'cache_size': 100, 'decision_function_shape': 'ovr', 'gamma': 'auto', 'kernel': 'rbf', 'probability': True, 'shrinking': True}
# 0.5920265780730897

y_predict = grid_search.predict(x_test)
print(classification_report(y_test, y_predict))
#               precision    recall  f1-score   support
#
#            0       0.77      0.80      0.78        99
#            1       0.61      0.56      0.58        55
#
#     accuracy                           0.71       154
#    macro avg       0.69      0.68      0.68       154
# weighted avg       0.71      0.71      0.71       154