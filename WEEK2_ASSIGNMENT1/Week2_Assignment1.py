import pandas as pd
df = pd.read_csv("C:\\Users\\minat\\OneDrive\\Documents\\Dataset 2.csv")

"""PART A: Dataset Understanding"""
# 1. Load the dataset and display the first 5 rows.
print(df.head())

# 2. Dtermine the number of rows and columns in the dataset.
print(df.shape)
print (f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")

# 3. Display all the column names in the dataset.
print(df.columns)

# 4. Identify the numerical and categorial features.
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns
print(f"Numerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")

# 5. Check whether the dataset contains any missing values.
missing_values = df.isnull().sum()
print(missing_values)


"""PART B: Exploratory Data Analysis"""
# 6. Calculate the average age of users.
average_age = df['Age'].mean()
print(f"Average age of users: {average_age}")

# 7. Determine the average watch hours per week.
average_watch_hours = df['WatchHoursPerWeek'].mean()
print(f"Average watch hours per week: {average_watch_hours}")

# 8. Find the average monthly spending of users.
average_monthly_spending = df['MonthlySpend'].mean()
print(f"Average monthly spending of users: {average_monthly_spending}")

# 9. Count the number of users in each subscription catergory.
subscription_counts = df['SubscriptionType'].value_counts()
print(f"Number of users in each subscription category:\n{subscription_counts}")

# 10. Determine the percentage of users who renewed their subscription.
renewal_percentage = (df['SubscriptionRenewed'].value_counts(normalize=True) * 100).get(1, 0)  # Assuming '1' indicates renewal
print(f"Percentage of users who renewed their subscription: {renewal_percentage}%")


"""PART C: Data Preparation"""
# 11. Convert categorial features into numerical form.
df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)
print(df_encoded.head())

# 12. Define the feature set(X) and the target variable (y) for subscription renewal prediction.
X = df_encoded.drop('SubscriptionRenewed', axis=1)
y = df_encoded['SubscriptionRenewed']
print(X.head())
print(y.head())

# 13. Split the dataset into training and testing sets (80% training, 20% testing).
import sklearn.model_selection as ms
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")   


"""PART D: Decision Tree Classification"""
# 14. Train a Decision Tree Model to predict whether a user will renew their subscription.
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 15. Evaluate the model using accuracy.
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# 16. Generate and interpret a confusion matrix.
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


"""PART E: K-Nearest Neighbors (KNN)"""
# 17. Train a KNN classifier with K=5.
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# 18. Compare the accuracy of the KNN model with the Decision Tree model.
y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Model accuracy: {accuracy_knn:.2f}")


"""PART F: Linear Regression"""
# 19. Train a Linear Regression model to predict monthly spending.
from sklearn.linear_model import LinearRegression
reg_model = LinearRegression()
reg_model.fit(X_train, df_encoded.loc[X_train.index, 'MonthlySpend'])

# 20. Predict the monthly spending for a new user and interpret the result.
new_user = pd.DataFrame({
    'Age': [30],
    'WatchHoursPerWeek': [10],
    'Gender_Male': [1],  # Assuming the new user is male
    'SubscriptionType_Basic': [0],
    'SubscriptionType_Premium': [1]
})
predicted_spending = reg_model.predict(new_user)
print(f"Predicted monthly spending for the new user: ${predicted_spending[0]:.2f}")

