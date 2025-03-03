
import pandas as pd
df=pd.read_csv(r"C:\Users\ELCOT\Desktop\Downloads\Iris_missingdata.csv")
df

x=df.iloc[:,1:5]

y=df.iloc[:,-1]



# Fill missing values with the mean of each column
i = x.fillna(x.mean())

print("\nDataset After Filling Null Values with Column Mean:")


print(df.isnull().sum().sum())

x=i

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
z = label_encoder.fit_transform(y)

y=z


from sklearn.preprocessing import StandardScaler

# Normalize the features using Standard Scaler
scaler = StandardScaler()
sd = pd.DataFrame(scaler.fit_transform(x))

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=47)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(r2)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
# Pairplot (Feature Relationships)
df['species'] = y  # Add species column back to the dataset for visualization
sns.pairplot(df, hue="species", diag_kind="kde", palette="husl")
plt.show()

from sklearn.metrics import classification_report
import pandas as pd
#classification report
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

plt.figure(figsize=(8, 5))
sns.heatmap(df_report.iloc[:-1, :].T, annot=True, cmap="coolwarm")
plt.title("Classification Report")
plt.show()




