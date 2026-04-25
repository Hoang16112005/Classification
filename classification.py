import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

data = pd.read_csv(r"diabetes.csv")

target = "Outcome"
x = data.drop(target, axis = 1) # loai bo cot outcome
y = data[target] # giu mot minh cot outcome

x_train, x_test, y_train, y_test = train_test_split(   #tách bộ test là 20% còn 80% dữ liệu
    x, y, test_size = 0.2, random_state = 42
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size = 0.25, random_state = 42 #tách bộ validate 20% từ 80% dữ liệu còn lại là 0,25%
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) #Bộ test không bao giờ được gọi fit

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print(classification_report(y_test, y_predict))
