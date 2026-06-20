import pandas as pd
df = pd.read_csv("C:\\Users\\minat\\OneDrive\\Documents\\agriculture_yield_dataset.csv")


"""Load the dataset and answer the following:"""

# PART A: UNDERSTANDING THE DATASET
""" Q1. Dataset Overview """
# ~ How many rows and columns are in the dataset?
print("Rows and Columns:", df.shape)
# ~ What are the names of all columns?
print("\nColumns Names:")
print(df.columns.tolist())
# ~ Display the first 10 records.
print("\nFirst 10 Records:")
print(df.head(10))

""" Q2. Data Types and Missing Values """
# ~ Check the data types of each column.
print("\nData Types:")
print(df.dtypes)
# ~ Identify whether there are any missing values in the dataset.
print("\nMissing Values:")
print(df.isnull().sum())
# ~ If missing values exist,mention the affected columns.
missing_cols = df.columns[df.isnull().sum() > 0]
if len(missing_cols) > 0:
    print("\nColumns with Missing Values:")
    print(missing_cols.tolist())
else:
    print("\nNo Missing Values Found")

""" Q3. Descriptive Statistics """
""" Generate summary statistics for all numerical features and answer: """
# ~ Which feature has the highest mean value?
print("\nSummary Statistics:")
print(df.describe())
numerical_cols = df.select_dtypes(include=['number']).columns
if len(numerical_cols) > 0:
    highest_mean_col = numerical_cols[df[numerical_cols].mean().idxmax()]
    print(f"\nFeature with Highest Mean Value: {highest_mean_col}")
# ~ Which feature has the highest standard deviation? 
if len(numerical_cols) > 0:
    highest_std_col = numerical_cols[df[numerical_cols].std().idxmax()]
    print(f"Feature with Highest Standard Deviation: {highest_std_col}")


# PART B: EXPLORATORY DATA ANALYSIS (EDA)
import matplotlib.pyplot as plt
""" Q4. Distribution Analysis """
# Create histograms for:  
# rainfall_mm  
# temperature_c  
# fertilizer_kg  
# yield_ton_per_hectare  
"""Write 2-3 observations from each histogram.""" 

plt.hist(df['rainfall_mm'], bins=20)
plt.title("Distribution of Rainfall")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Frequency")
plt.show()

plt.hist(df['temperature_c'], bins=20)
plt.title("Distribution of Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.show()

plt.hist(df['fertilizer_kg'], bins=20)
plt.title("Distribution of Fertilizer Usage")
plt.xlabel("Fertilizer (kg)")
plt.ylabel("Frequency")
plt.show()

plt.hist(df['yield_ton_per_hectare'], bins=20)
plt.title("Distribution of Crop Yield")
plt.xlabel("Yield (ton/hectare)")
plt.ylabel("Frequency")
plt.show()

""" Q5. Crop Type Analysis """ 
# Find the number of records for each crop type.  
# Create a count plot (bar chart) for crop_type.  
# Which crop appears most frequently?

crop_counts = df['crop_type'].value_counts()
print(crop_counts)

crop_counts.plot(kind='bar')
plt.title("Crop Type Count")
plt.xlabel("Crop Type")
plt.ylabel("Count")
plt.show()

most_frequent_crop = df['crop_type'].value_counts().idxmax()
count = df['crop_type'].value_counts().max()
print("Most Frequent Crop:", most_frequent_crop)
print("Count:", count)

""" Q6. Soil Type Analysis """
# Find the frequency of each soil type.
# Create a count plot for soil_type.
# Which soil type is most common?

soil_counts = df['soil_type'].value_counts()
print(soil_counts)

df['soil_type'].value_counts().plot(kind='bar')
plt.title("Soil Type Distribution")
plt.xlabel("Soil Type")
plt.ylabel("Count")
plt.show()

most_common_soil = df['soil_type'].value_counts().idxmax()
print("Most Common Soil Type:", most_common_soil)

""" Q7. Yield Distribution """
# Create a histogram of yield_ton_per_hectare.
# Is the distribution approximately normal?
# Are there any noticeable outliers?

plt.hist(df['yield_ton_per_hectare'], bins=20)
plt.title("Yield Distribution")
plt.xlabel("Yield (ton/hectare)")
plt.ylabel("Frequency")
plt.show()

print(df['yield_ton_per_hectare'].describe())

Q1 = df['yield_ton_per_hectare'].quantile(0.25)
Q3 = df['yield_ton_per_hectare'].quantile(0.75)

IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['yield_ton_per_hectare'] < lower) |
              (df['yield_ton_per_hectare'] > upper)]
print("Number of Outliers:", len(outliers))

""" Q8. Scatter Plot Analysis """
# Create scatter plots of:
# 1. rainfall_mm vs yield_ton_per_hectare
# 2. fertilizer_kg vs yield_ton_per_hectare
# Based on the plots:
# Which feature appears to have a stronger relationship with yield?

plt.scatter(df['rainfall_mm'],
            df['yield_ton_per_hectare'])
plt.xlabel("Rainfall (mm)")
plt.ylabel("Yield (ton/hectare)")
plt.title("Rainfall vs Yield")
plt.show()

plt.scatter(df['fertilizer_kg'],
            df['yield_ton_per_hectare'])
plt.xlabel("Fertilizer (kg)")
plt.ylabel("Yield (ton/hectare)")
plt.title("Fertilizer vs Yield")
plt.show()

corr_rainfall = df['rainfall_mm'].corr(df['yield_ton_per_hectare'])
corr_fertilizer = df['fertilizer_kg'].corr(df['yield_ton_per_hectare'])

print("Rainfall Correlation:", corr_rainfall)
print("Fertilizer Correlation:", corr_fertilizer)

if abs(corr_rainfall) > abs(corr_fertilizer):
    print("Rainfall has stronger relationship with yield.")
else:
    print("Fertilizer has stronger relationship with yield.")

""" Q9. Correlation Analysis """ 
# Generate a correlation matrix for numerical features.  
# Create a heatmap.  
# Identify the top three features most correlated with crop yield. 

correlation_matrix = df.corr(numeric_only=True)
print(correlation_matrix)

import seaborn as sns

corr = df.corr(numeric_only=True)
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

yield_corr = corr['yield_ton_per_hectare'].drop('yield_ton_per_hectare')
top3 = yield_corr.abs().sort_values(ascending=False).head(3)
print(top3)

""" Q10. Group-Based Analysis """ 
# Calculate the average yield for: 
# Each crop type  
# Each soil type  
# Which crop and soil type have the highest average yield?

avg_crop_yield = df.groupby('crop_type')['yield_ton_per_hectare'].mean()
print(avg_crop_yield)

avg_soil_yield = df.groupby('soil_type')['yield_ton_per_hectare'].mean()
print(avg_soil_yield)

crop_yield = df.groupby('crop_type')['yield_ton_per_hectare'].mean()
soil_yield = df.groupby('soil_type')['yield_ton_per_hectare'].mean()
print("Best Crop:", crop_yield.idxmax())
print("Highest Crop Yield:", crop_yield.max())
print("Best Soil Type:", soil_yield.idxmax())
print("Highest Soil Yield:", soil_yield.max())


# PART C: DATA PREPARATION
""" Q11. Feature Encoding """
# The dataset contains categorical variables. 
# Identify the categorical columns.  
# Convert them into numerical form using One-Hot Encoding.  
# Display the first five rows of the transformed dataset.

categorical_cols = df.select_dtypes(include='object').columns
print(categorical_cols)

encoded_df = pd.get_dummies(df, drop_first=True)
print(encoded_df.head())

encoded_df = pd.get_dummies(df, drop_first=True)
print(encoded_df.head(5))

""" Q12. Feature Selection """ 
# Separate: 
# Input features (X)  
# Target variable (y)  
# Specify which column is being used as the target variable. 

df = pd.get_dummies(df, drop_first=True)

X = df.drop('yield_ton_per_hectare', axis=1)
y = df['yield_ton_per_hectare']

print("Target Variable: yield_ton_per_hectare")
print("X Shape:", X.shape)
print("y Shape:", y.shape)


# PART D: MACHINE LEARNING
""" Q13. Train-Test Split """
# Split the dataset into: 
# 80% Training Data  
# Display the shape of:
# X_train  
# X_test  
# y_train  
# y_test 

from sklearn.model_selection import train_test_split

df = pd.get_dummies(df, drop_first=True)

X = df.drop('yield_ton_per_hectare', axis=1)
y = df['yield_ton_per_hectare']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

""" Q14. Linear Regression Model """ 
# Train a Linear Regression model.  
# Display the model coefficients and intercept.  
# Which feature has the highest positive coefficient?

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.get_dummies(df, drop_first=True)
X = df.drop('yield_ton_per_hectare', axis=1)
y = df['yield_ton_per_hectare']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
print("Model Trained Successfully")

df = pd.get_dummies(df, drop_first=True)
X = df.drop('yield_ton_per_hectare', axis=1)
y = df['yield_ton_per_hectare']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
print("Intercept:")
print(model.intercept_)
print("\nCoefficients:")
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print(coef_df)

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
highest_feature = coef_df.loc[
    coef_df['Coefficient'].idxmax()
]
print(highest_feature)
