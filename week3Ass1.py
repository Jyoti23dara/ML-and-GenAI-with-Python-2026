# ==========================================================
# Exploratory Data Analysis (EDA) and Machine Learning
# on Agricultural Yield Dataset
# ==========================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Load Dataset


df = pd.read_csv("C:\\Users\\3410p\\Downloads\\agriculture_yield_dataset.csv")

# Q1. Dataset Overview


print("\n Q1. Dataset Overview ")

print("Number of Rows and Columns:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nFirst 10 Records:")
print(df.head(10))

# Q2. Data Types and Missing Values


print("\n========== Q2. Data Types and Missing Values ==========")

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())


# Q3. Descriptive Statistics


print("\n========== Q3. Descriptive Statistics ==========")

print(df.describe())

print("\nMean Values:")
print(df.mean(numeric_only=True))

print("\nStandard Deviation:")
print(df.std(numeric_only=True))


# Q4. Distribution Analysis


print("\n========== Q4. Distribution Analysis ==========")

plt.figure(figsize=(6,4))
plt.hist(df["rainfall_mm"], bins=20)
plt.title("Rainfall Distribution")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df["temperature_c"], bins=20)
plt.title("Temperature Distribution")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df["fertilizer_kg"], bins=20)
plt.title("Fertilizer Distribution")
plt.xlabel("Fertilizer (kg)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df["yield_ton_per_hectare"], bins=20)
plt.title("Yield Distribution")
plt.xlabel("Yield")
plt.ylabel("Frequency")
plt.show()


# Q5. Crop Type Analysis


print("\n========== Q5. Crop Type Analysis ==========")

print(df["crop_type"].value_counts())

plt.figure(figsize=(7,4))
sns.countplot(x="crop_type", data=df)
plt.title("Crop Type Count")
plt.show()


# Q6. Soil Type Analysis


print("\n========== Q6. Soil Type Analysis ==========")

print(df["soil_type"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x="soil_type", data=df)
plt.title("Soil Type Count")
plt.show()


# Q7. Yield Distribution


print("\n========== Q7. Yield Distribution ==========")

plt.figure(figsize=(6,4))
plt.hist(df["yield_ton_per_hectare"], bins=20)
plt.title("Yield Distribution")
plt.xlabel("Yield")
plt.ylabel("Frequency")
plt.show()


# Q8. Scatter Plot Analysis


print("\n========== Q8. Scatter Plot Analysis ==========")

plt.figure(figsize=(6,4))
plt.scatter(df["rainfall_mm"],
            df["yield_ton_per_hectare"])

plt.xlabel("Rainfall (mm)")
plt.ylabel("Yield")
plt.title("Rainfall vs Yield")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df["fertilizer_kg"],
            df["yield_ton_per_hectare"])

plt.xlabel("Fertilizer (kg)")
plt.ylabel("Yield")
plt.title("Fertilizer vs Yield")
plt.show()


# Q9. Correlation Analysis


print("\n========== Q9. Correlation Analysis ==========")

corr_matrix = df.select_dtypes(include=np.number).corr()

print(corr_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix,
            annot=True,
            cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.show()


# Q10. Group-Based Analysis

print("\n========== Q10. Group-Based Analysis ==========")

crop_yield = df.groupby(
    "crop_type"
)["yield_ton_per_hectare"].mean()

print("\nAverage Yield by Crop Type:")
print(crop_yield)

soil_yield = df.groupby(
    "soil_type"
)["yield_ton_per_hectare"].mean()

print("\nAverage Yield by Soil Type:")
print(soil_yield)


# Q11. Feature Encoding

print("\n========== Q11. Feature Encoding ==========")

encoded_df = pd.get_dummies(
    df,
    columns=["crop_type", "soil_type"]
)

print(encoded_df.head())


# Q12. Feature Selection


print("\n========== Q12. Feature Selection ==========")

X = encoded_df.drop(
    "yield_ton_per_hectare",
    axis=1
)

y = encoded_df["yield_ton_per_hectare"]

print("X Shape:", X.shape)
print("y Shape:", y.shape)


# # Q13. Train-Test Split


print("\n========== Q13. Train-Test Split ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)


# # Q14. Linear Regression Model


print("\n========== Q14. Linear Regression Model ==========")

model = LinearRegression()

model.fit(X_train, y_train)

print("\nIntercept:")
print(model.intercept_)

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nModel Coefficients:")
print(coefficients)

highest_coef = coefficients.loc[
    coefficients["Coefficient"].idxmax()
]

print("\nFeature with Highest Positive Coefficient:")
print(highest_coef)