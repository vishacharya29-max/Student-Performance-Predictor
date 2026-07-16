import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

#loading dataset
df = pd.read_csv("student_dataset.csv")

#Features and Target
X = df[["Attendance","Internal_Marks"]]
y = df["Result"]

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

#Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved successfully.")

