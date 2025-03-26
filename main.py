import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("dataset/diabetes.csv")
profile = ProfileReport(data, title="Diabetes Report", explorative=True)
profile.to_file("report/report_statistics.html")

# data split
target = "Outcome"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# data processing
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# performance testing on multiple models
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)
print(models)
#>>> LogisticRegression > RandomForest > SVC
