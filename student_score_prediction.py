# student score prediction
# This project predicts exam scores based on study hours.
import pandas as pd
import matplolib.pylot as plt
from sklearn.model_selection import train_test-split
from sklearn.Linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
# step 1 : Create a dataset
data = { 
    "Hours":[ 
        2.5, 5.1, 3.2, 8.5, 3.5, 
        1.5, 9.2, 5.5, 8.3, 2.7,
        7.7, 5.9, 4.5, 3.3, 1.1,
        8.9, 2.5, 1.9, 6.1, 7.4
    ],
   "Scores":[ 
       21, 47, 27, 75, 30,
       20, 88, 60, 81, 25,
       85, 62, 41, 42, 17,
       95, 30, 24, 67, 69
   ]
}
# Step 2:  Convert the data into a DataFrame
df = pd.DataFrame(data)
# Step 3: Select input and target
X = df[["Hours"]]
y = df [["Scores"]]
# Step 4 : Slipt the data
X_train, X_test, y_train, y_test= train_test_split( X, y, test_size=0.2, random_state=42)
# Step 5: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Step 6 : Predict scores for test data
prediction = model.predict( X_test)
# Step 7 : Evaluate the model 
mae = mean_absolute_error( y_test, predictions)
print("Mean Absolute Error:", round (mae,2))
# Step 8: Predict a score for a new student 
hours = float(input("Enter hours studied:"))
new_student = pd.DataFrame({"Hours":[hours]})
predicted_score = model.predict(new_student)[0]
Print ("Predicted exam score:", round(predicted_score,2))
# Step 9; Display a graph
plt.scatter(df["Hours"], df["Scores"], color= "blue")
plt.plot(df["Hours"], model.predict(X), color="red")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Student Score Prediction")
plt.show()
