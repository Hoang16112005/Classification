import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from lazypredict.Supervised import LazyClassifier
import pickle

data = pd.read_csv(r"diabetes.csv")

target = "Outcome"
x = data.drop(target, axis = 1) # loai bo cot outcome
y = data[target] # giu mot minh cot outcome

x_train, x_test, y_train, y_test = train_test_split(   #tách bộ test là 20% còn 80% dữ liệu
    x, y, test_size = 0.2, random_state = 42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) #Bộ test không bao giờ được gọi fit

# params = {
#     "n_estimators": [50, 100, 200],
#     "criterion": ["gini", "entropy","log_loss"]
# }
#
# model = GridSearchCV(RandomForestClassifier(random_state = 100), param_grid = params, scoring = "recall", cv=6, verbose = 2)
# model.fit(x_train, y_train)
# y_predict = model.predict(x_test)
# print(model.best_params_)
# print(model.best_score_)
# print(classification_report(y_test, y_predict))

clf = LazyClassifier(verbose = 0, ignore_warnings = True, custom_metric=None)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)