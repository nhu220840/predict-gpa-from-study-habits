# Import necessary libraries
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport  # For generating data profile reports
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MultiLabelBinarizer, OneHotEncoder  # For data preprocessing
from sklearn.compose import ColumnTransformer  # For applying different transformations to different columns
from sklearn.ensemble import RandomForestRegressor  # Machine learning algorithm
from sklearn.pipeline import Pipeline  # For creating a processing pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin  # Base classes for custom estimators
from sklearn.linear_model import LinearRegression  # Linear regression algorithm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # Model evaluation metrics


# Custom transformer for handling multi-label categorical features
class MultiLabelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()  # Initialize multi-label binarizer

    def fit(self, X, y=None):
        # Convert string of comma-separated values into lists
        X_series = pd.Series(X.squeeze())
        X_series = X_series.apply(lambda x: x.split(', ') if isinstance(x, str) else x)
        self.mlb.fit(X_series)  # Fit the binarizer
        return self

    def transform(self, X):
        # Transform the data using the fitted binarizer
        X_series = pd.Series(X.squeeze())
        X_series = X_series.apply(lambda x: x.split(', ') if isinstance(x, str) else x)
        return self.mlb.transform(X_series)

# Custom regressor that bounds predictions within a specified range
class BoundedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor, min_val=0, max_val=10):
        self.regressor = regressor  # Base regressor
        self.min_val = min_val  # Minimum allowed value for predictions
        self.max_val = max_val  # Maximum allowed value for predictions

    def fit(self, X, y):
        # Fit the base regressor
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        # Make predictions and clip them to the specified range
        predictions = self.regressor.predict(X)
        return np.clip(predictions, self.min_val, self.max_val)

# Data loading and initial processing
data = pd.read_csv('gpa-collections.csv')  # Load the dataset
profile = ProfileReport(data, title="Student Score Report")  # Generate a profile report
# profile.to_file('report_csv.html')  # Save the report (commented out)

# Define the target variable
target = "Latest semester GPA"
data.columns = data.columns.str.strip()  # Remove whitespace from column names

# Split data into features and target
x = data.drop(columns=[target, "Timestamp"], axis=1)  # Features
y = data[target]  # Target variable

# Split data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define feature groups based on their data types and characteristics
# Numerical features
num_features = ["Latest academic year GPA"]

# Ordinal features (categorical with inherent order)
ord_features = [
    "Current education level",
    "Daily study duration",
    "Study break frequency",
    "Exam preparation approach",
    "Daily sleep duration",
    "Weekly exercise frequency",
    "Social media usage during study time",
    "Gender",
    "Use of technology in studying",
    "Lack of focus in studying",
    "Phone usage while studying"
]

# Multi-label nominal features (categorical with multiple possible values per entry)
multi_nom_features = ["Preferred study method"]

# Single-label nominal features (categorical without inherent order)
single_nom_features = [
    "Most effective study time",
    "Dietary Habits",
    "Preferred study environment"
]

# Define the categories for ordinal features to maintain their order
education_levels = ["Middle school", "High school", "University", "Graduated"]
daily_study_duration = ["< 1 hour", "1-2 hours", "2-4 hours", "> 4 hours"]
study_break_freq = ["Almost no breaks", "5 min/hour", "10-15 min/hour", "15+ min/hour"]
exam_preparation = ["No preparation", "1 day before", "1-2 weeks before", "> 2 weeks before"]
daily_sleep_duration = ["< 4 hours", "4-6 hours", "6-8 hours", "> 8 hours"]
weekly_exercise_freq = ["Never", "1-2 times", "3-4 times", "5+ times"]
social_media_usage = ["Never", "< 15 min/hour", "15-30 min/hour", "> 30 min/hour"]
gender = ["Male", "Female"]
use_technology = ["Yes", "No"]
focus = ["Yes", "No"]
phone_usage = ["Yes", "No"]

# Define categories for single-label nominal features
effective_study_time = ["Morning", "Afternoon", "Evening", "Late night"]
diet_habits = ["No specific diet", "Balanced diet", "Special diet (Vegan, Keto, etc.)"]
study_env = ["Silent", "Soft background music", "Noisy (cafe, crowded library, etc.)"]

# Pipeline for numerical features: impute missing values and standardize
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),  # Fill missing values with median
    ("scaler", StandardScaler())  # Standardize features to have mean=0 and variance=1
])

# Pipeline for ordinal features: impute missing values and encode with ordered categories
ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing values with most frequent value
    ("encoder", OrdinalEncoder(categories=[
        education_levels,
        daily_study_duration,
        study_break_freq,
        exam_preparation,
        daily_sleep_duration,
        weekly_exercise_freq,
        social_media_usage,
        gender,
        use_technology,
        focus,
        phone_usage,
    ]))  # Encode ordinal categories while preserving order
])

# Pipeline for multi-label nominal features: impute missing values and use custom transformer
multi_nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing values with most frequent value
    ("encoder", MultiLabelTransformer())  # Apply custom multi-label transformation
])

# Pipeline for single-label nominal features: impute missing values and one-hot encode
single_nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing values with most frequent value
    ("encoder", OneHotEncoder(categories=[
        effective_study_time,
        diet_habits,
        study_env,
    ], handle_unknown="ignore"))  # One-hot encode nominal categories
])

# Combine all feature transformers using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, num_features),
    ("ord_feature", ord_transformer, ord_features),
    ("multi_nom_feature", multi_nom_transformer, multi_nom_features),
    ("single_nom_feature", single_nom_transformer, single_nom_features),
])

# Create the full pipeline: preprocessing followed by the bounded linear regression model
reg = Pipeline(steps=[
    ("preprocessor", preprocessor),  # Apply feature preprocessing
    ("model", BoundedRegressor(LinearRegression(), min_val=0, max_val=10))  # Apply bounded linear regression (GPA between 0-10)
])

# Train the model
reg.fit(x_train, y_train)

# Make predictions on the test set
y_predict = reg.predict(x_test)

# Print predicted vs actual values
for i, j in zip(y_predict, y_test):
    print(f"Predicted value: {i:.2f}. Actual value: {j}")

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

# Print evaluation metrics
print(f"\nEvaluate model:")
print(f"MAE: {mean_absolute_error(y_test, y_predict)}")  # Mean Absolute Error
print(f"MSE: {mean_squared_error(y_test, y_predict)}")   # Mean Squared Error
print(f"R2: {r2_score(y_test, y_predict)}")              # R-squared (coefficient of determination)