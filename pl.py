import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score #for cross validation
import seaborn as sns
import matplotlib.pyplot as plt



# Load the data from CSV
data = pd.read_csv(r"C:\Users\Admin\Desktop\PLACEMENTPRO\PLACEMENTPRO.csv")

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the first few rows
print(df.head())

# Encode categorical features if necessary
le = LabelEncoder()
df['Stream'] = le.fit_transform(df['Stream'])

# Define the target variable based on the given criteria
df['Placed'] = ((df['Backlogs'] == 0) & (df['CGPA'] > 7)).astype(int)

# Define features and target variable
X = df[['Backlogs', 'Stream', 'CGPA']]
y = df['Placed']



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

 

#manual checking of data
def check_eligibility(backlogs, cgpa):
    return 1 if backlogs == 0 and cgpa > 7 else 0

print(check_eligibility(0, 8.0))  # Output: 1 (Eligible)
print(check_eligibility(1, 8.0))  # Output: 0 (Not Eligible)



# Save the model
with open('logistic_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)



#testing with new data
new_data = pd.DataFrame({
    'Backlogs': [0, 1],
    'Stream': [1, 2],
    'CGPA': [8.0, 6.5]
})

new_predictions = model.predict(new_data)
print(new_predictions)



# Plot distributions
sns.histplot(df['CGPA'])
plt.title('CGPA Distribution')
plt.show()




































