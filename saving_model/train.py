import pandas as pd  # pip install pandas
from sklearn import model_selection # pip install scikit-learn
from sklearn.linear_model import LogisticRegression
import joblib
import pickle
import numpy as np

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df = pd.read_csv(r'C:\Users\Naveen Reddy\Downloads\diabetes.csv')
# df.columns= ['preg','plas','pres','skin','test','mass','pedi','age','class']

X = df.iloc[: , :8]
y = df.iloc[:, 8]

#X = np.array(X.fit_transform(X), dtype=np.float64)
df.info()

X_train , X_test, y_train, y_test = model_selection.train_test_split(X , y , test_size = 0.2 , random_state = 101)

# train the model
model = LogisticRegression()
model.fit(X_train , y_train)

# accuracy
result = model.score(X_test , y_test)

print(result)


# save the model (.sav , / .pkl)
joblib.dump(model , 'dib.pkl')

