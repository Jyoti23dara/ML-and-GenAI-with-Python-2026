# Imort Reaquired  libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score, confusion_matrix 

# Part A: Dataset Understanding
# Q1. Load the dataset and display the first five records.

df = pd.read_csv("C:/Users/3410p/Downloads/Dataset2.csv")

print("First Five Records:")
print(df.head())
# Q2. Determine the number of rows and columns in the dataset.
rows, columns = df.shape

print("Number of Rows:", rows)
print("Number of Columns:", columns)

# Q3. Display all column names.
print("Column Names:")
print(df.columns)

# Q4. Identify numerical and categorical features.
print("Numerical Features:")
print(df.select_dtypes(include=['int64', 'float64']).columns)

print("\nCategorical Features:")
print(df.select_dtypes(include=['object']).columns) 

# Q5. Check whether the dataset contains missing values.
print("Missing Values:")
print(df.isnull().sum())

# Part B: Exploratory Data Analysis
# Q6. Calculate the average age of users.

average_age = df['Age'].mean()

print("Average Age of Users:", average_age)

# Q7. Determine the average watch hours per week.
average_watch_hours = df['WatchHoursPerWeek'].mean()

print("Average Watch Hours Per Week:", average_watch_hours)

# Q8. Find the average monthly spending of users.
average_spending = df['MonthlySpend'].mean()

print("Average Monthly Spending:", average_spending)

# Q9. Count the number of users in each subscription category.
subscription_count = df['SubscriptionType'].value_counts()

print("Users in Each Subscription Category:")
print(subscription_count)

# Q10. Determine the percentage of users who renewed their subscriptions.
renewal_percentage = (
    (df['SubscriptionRenewed'] == 'Yes').sum()
    / len(df)
) * 100

print("Subscription Renewal Percentage:", renewal_percentage)

# Part C: Data Preparation
# Q11. Convert categorical features into numerical form.
label_encoder = LabelEncoder()

categorical_columns = [
    'Gender',
    'SubscriptionType',
    'FavoriteGenre',
    'SubscriptionRenewed'
]

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

print(df.head())

# Q12. Define feature set (X) and target variable (y)
X = df.drop(['UserID', 'SubscriptionRenewed'], axis=1)

y = df['SubscriptionRenewed']

print("Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())

# Q13. Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Set Size:", X_train.shape)
print("Testing Set Size:", X_test.shape)

# Part D: Decision Tree Classification
# Q14. Train a Decision Tree model.

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

print("Decision Tree Model Trained Successfully")

# Q15. Evaluate the model using accuracy.
dt_predictions = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_predictions)

print("Decision Tree Accuracy:", dt_accuracy)

# Q16. Generate and interpret the confusion matrix.
cm = confusion_matrix(y_test, dt_predictions)

print("Confusion Matrix:")
print(cm)

# Part E: K-Nearest Neighbors (KNN)
# Q17. Train a KNN classifier with K = 5.
knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

knn_predictions = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_predictions)

print("KNN Accuracy:", knn_accuracy)

# Q18. Compare the accuracy of KNN with Decision Tree.
print("Decision Tree Accuracy:", dt_accuracy)
print("KNN Accuracy:", knn_accuracy)

if dt_accuracy > knn_accuracy:
    print("Decision Tree performed better.")
elif knn_accuracy > dt_accuracy:
    print("KNN performed better.")
else:
    print("Both models performed equally.")

# Part F: Linear Regression
# Q19. Train a Linear Regression model to predict monthly spending.
X_reg = df.drop(['UserID', 'MonthlySpend'], axis=1)

y_reg = df['MonthlySpend']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

lr_model = LinearRegression()

lr_model.fit(X_train_reg, y_train_reg)

print("Linear Regression Model Trained Successfully")

# Q20. Predict monthly spending for a new user.
new_user = [[
    25,     # Age
    1,      # Gender
    2,      # SubscriptionType
    15,     # WatchHoursPerWeek
    3,      # DevicesUsed
    1,      # FavoriteGenre
    10,     # AdClicks
    1       # SubscriptionRenewed
]]

predicted_spending = lr_model.predict(new_user)

print("Predicted Monthly Spending:", predicted_spending[0])
