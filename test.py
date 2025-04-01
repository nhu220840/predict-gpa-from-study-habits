import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class MultiLabelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
        X_series = pd.Series(X.squeeze())
        X_series = X_series.apply(lambda x: x.split(', ') if isinstance(x, str) else x)
        self.mlb.fit(X_series)
        return self

    def transform(self, X):
        X_series = pd.Series(X.squeeze())
        X_series = X_series.apply(lambda x: x.split(', ') if isinstance(x, str) else x)
        return self.mlb.transform(X_series)

class BoundedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor, min_val=0, max_val=10):
        self.regressor = regressor
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, X, y):
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        predictions = self.regressor.predict(X)
        return np.clip(predictions, self.min_val, self.max_val)

data = pd.read_csv('gpa-collections.csv')
profile = ProfileReport(data, title="Student Score Report")
# profile.to_file('report_csv.html')  # Lưu báo cáo nếu cần

target = "Latest semester GPA"
data.columns = data.columns.str.strip()

x = data.drop(columns=[target, "Timestamp"], axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

num_features = ["Latest academic year GPA"]
ord_features = [
    "Current education level",
    "Daily study duration",
    "Study break frequency",
    "Exam preparation approach",
    "Daily sleep duration",
    "Weekly exercise frequency",
    "Social media usage during study time"
]
nom_features = ["Preferred study method"]

education_levels = ["Middle school", "High school", "University", "Graduated"]
daily_study_duration = ["< 1 hour", "1-2 hours", "2-4 hours", "> 4 hours"]
study_break_freq = ["Almost no breaks", "5 min/hour", "10-15 min/hour", "15+ min/hour"]
exam_preparation = ["No preparation", "1 day before", "1-2 weeks before", "> 2 weeks before"]
daily_sleep_duration = ["< 4 hours", "4-6 hours", "6-8 hours", "> 8 hours"]
weekly_exercise_freq = ["Never", "1-2 times", "3-4 times", "5+ times"]
social_media_usage = ["None", "< 15 min/hour", "15-30 min/hour", "> 30 min/hour"]

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[
        education_levels,
        daily_study_duration,
        study_break_freq,
        exam_preparation,
        daily_sleep_duration,
        weekly_exercise_freq,
        social_media_usage
    ]))
])

nom_transformer = MultiLabelTransformer()

preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, num_features),
    ("ord_feature", ord_transformer, ord_features),
    ("nom_feature", nom_transformer, nom_features),
])

reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", BoundedRegressor(LinearRegression(), min_val=0, max_val=10))
])

reg.fit(x_train, y_train)

y_predict = reg.predict(x_test)

for i, j in zip(y_predict, y_test):
    print(f"Predicted value: {i:.2f}. Actual value: {j}")

mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(f"\nEvaluate model:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")