import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import warnings
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore", message="X does not have valid feature names, but SVC was fitted with feature names")

df = pd.read_csv("")
label_encoder = LabelEncoder()
df['LUNG_CANCER'] = label_encoder.fit_transform(df['LUNG_CANCER'])
df['GENDER'] = label_encoder.fit_transform(df['GENDER'])
df = df.drop('AGE', axis=1)
X = df.drop('LUNG_CANCER', axis=1)
Y = df['LUNG_CANCER']
x_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = SVC(probability=True)
model.fit(x_train, y_train)
print("Enter 1 if you have the symptom, and 0 if you don't.")
symptoms = []
for feature in X.columns:
    symptom_input = input(f"{feature}: ")
    while symptom_input not in ["1", "0"]:
        print("Invalid input. Please enter 1 or 0.")
        symptom_input = input(f"{feature}: ")
    symptoms.append(int(symptom_input))
predict=model.predict([symptoms])
probab=model.predict_proba([symptoms])
if predict[0] == 1:
    print("The model predicts that the person has lung cancer.")
else:
    print("The model predicts that the person does not have lung cancer.")

print("Prediction Probability:", probab)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
