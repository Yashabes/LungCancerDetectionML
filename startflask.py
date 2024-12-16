import pandas as pd
from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv(r"C:\Users\taliy\OneDrive\Desktop\DATA3.csv")
df['GENDER'] = df['GENDER'].map({'M': 0, 'F': 1})
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'NO': 0, 'YES': 1})
X = df.drop(columns=['LUNG_CANCER'])  # Features
y = df['LUNG_CANCER']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=8)  
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        GENDER = request.form['gender']
        AGE = request.form['age']
        SMOKING = request.form['smoking']
        YELLOW_FINGERS = request.form['yellow_fingers']
        ANXIETY = request.form['anxiety']
        PEER_PRESSURE = request.form['peer_pressure']
        CHRONIC_DISEASE = request.form['chronic_disease']
        FATIGUE = request.form['fatigue']
        ALLERGY = request.form['allergy']
        WHEEZING = request.form['wheezing']
        ALCOHOL_CONSUMING = request.form['alcohol']
        COUGHING = request.form['coughing']
        SHORTNESS_OF_BREATH = request.form['shortness_of_breath']
        SWALLOWING_DIFFICULTY = request.form['swallowing_difficulty']
        CHEST_PAIN = request.form['chest_pain']

        GENDER = 0 if GENDER == "M" else 1
        SMOKING = 1 if SMOKING == "Y" else 0
        YELLOW_FINGERS = 1 if YELLOW_FINGERS == "Y" else 0
        ANXIETY = 1 if ANXIETY == "Y" else 0
        PEER_PRESSURE = 1 if PEER_PRESSURE== "Y" else 0
        CHRONIC_DISEASE = 1 if CHRONIC_DISEASE == "Y" else 0
        FATIGUE = 1 if FATIGUE == "Y" else 0
        ALLERGY = 1 if ALLERGY == "Y" else 0
        WHEEZING = 1 if WHEEZING == "Y" else 0
        ALCOHOL_CONSUMING = 1 if ALCOHOL_CONSUMING == "Y" else 0
        COUGHING = 1 if COUGHING == "Y" else 0
        SHORTNESS_OF_BREATH = 1 if SHORTNESS_OF_BREATH == "Y" else 0
        SWALLOWING_DIFFICULTY = 1 if SWALLOWING_DIFFICULTY == "Y" else 0
        CHEST_PAIN = 1 if CHEST_PAIN == "Y" else 0

        input_data = pd.DataFrame([[
        GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE,
        FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING,SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN
        ]], columns=X.columns)

        input_scaled = scaler.transform(input_data)
        prediction = knn.predict(input_scaled)

        result = "LUNG CANCER: YES" if prediction[0] == 1 else "LUNG CANCER:NO"
        #print(f"Lung cancer prediction: {result}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return render_template('SITE.html', result=result)
    return render_template('SITE.html', result=None)
if __name__ == '__main__':
    app.run(debug=True)
