import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MultiLabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class MultiLabelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        self.is_fitted_ = False  # Thêm thuộc tính để kiểm tra trạng thái

    def fit(self, X, y=None):
        # Kiểm tra X phải là mảng hoặc DataFrame
        if not isinstance(X, (np.ndarray, pd.DataFrame, pd.Series)):
            raise ValueError("Input phải là numpy array hoặc pandas DataFrame/Series")

        # Xử lý đầu vào một cách nhất quán
        if isinstance(X, pd.DataFrame):
            X_series = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            X_series = X
        else:
            X_series = pd.Series(X.squeeze())

        # Đảm bảo xử lý đúng cách với các chuỗi
        X_processed = X_series.apply(lambda x: x.split(', ') if isinstance(x, str) else
        [] if pd.isna(x) else [str(x)])

        self.mlb.fit(X_processed)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        # Kiểm tra transformer đã được fit chưa
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("MultiLabelTransformer chưa được fit trước khi transform")

        # Xử lý đầu vào giống như phương thức fit
        if isinstance(X, pd.DataFrame):
            X_series = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            X_series = X
        else:
            X_series = pd.Series(X.squeeze())

        X_processed = X_series.apply(lambda x: x.split(', ') if isinstance(x, str) else
        [] if pd.isna(x) else [str(x)])

        return self.mlb.transform(X_processed)


class BoundedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor, min_val=0, max_val=10):
        self.regressor = regressor
        self.min_val = min_val
        self.max_val = max_val
        self.is_fitted_ = False

    def fit(self, X, y):
        self.regressor.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("BoundedRegressor chưa được fit trước khi predict")
        predictions = self.regressor.predict(X)
        return np.clip(predictions, self.min_val, self.max_val)


# Tải dữ liệu
print("Đang tải dữ liệu...")
try:
    data = pd.read_csv('gpa-collections.csv')
    print(f"Đã tải dữ liệu. Kích thước: {data.shape}")
except Exception as e:
    print(f"Lỗi khi tải dữ liệu: {e}")
    raise

# Báo cáo dữ liệu
print("Đang tạo báo cáo dữ liệu...")
profile = ProfileReport(data, title="Student Score Report", minimal=True)
# profile.to_file('report_csv.html')

# Định nghĩa biến mục tiêu và features
print("Đang chuẩn bị dữ liệu...")
target = "Latest semester GPA"
data.columns = data.columns.str.strip()

# Kiểm tra giá trị NaN
missing_values = data.isnull().sum()
print(f"Số giá trị thiếu trong từng cột:\n{missing_values}")

# In một số thông tin về dữ liệu
print(f"Thông tin về dữ liệu:\n{data.info()}")
print(f"Thống kê mô tả:\n{data.describe()}")

# Tách dữ liệu thành features và target
x = data.drop(columns=[target, "Timestamp"], axis=1)
y = data[target]

# Chia dữ liệu thành tập huấn luyện và kiểm thử
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"Kích thước tập huấn luyện: {x_train.shape}, {y_train.shape}")
print(f"Kích thước tập kiểm thử: {x_test.shape}, {y_test.shape}")

# Định nghĩa nhóm features
num_features = ["Latest academic year GPA"]
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
multi_nom_features = ["Preferred study method"]
single_nom_features = [
    "Most effective study time",
    "Dietary Habits",
    "Preferred study environment"
]

# Kiểm tra xem các cột có tồn tại trong dữ liệu
all_features = num_features + ord_features + multi_nom_features + single_nom_features
missing_cols = set(all_features) - set(x.columns)
if missing_cols:
    print(f"CẢNH BÁO: Các cột sau không có trong dữ liệu: {missing_cols}")

# Định nghĩa các danh mục cho features thứ tự
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

# Kiểm tra giá trị duy nhất trong mỗi cột
for feature in ord_features:
    if feature in x.columns:
        unique_vals = x[feature].unique()
        print(f"Giá trị duy nhất trong {feature}: {unique_vals}")

# Định nghĩa các danh mục cho features danh định
effective_study_time = ["Morning", "Afternoon", "Evening", "Late night"]
diet_habits = ["No specific diet", "Balanced diet", "Special diet (Vegan, Keto, etc.)"]
study_env = ["Silent", "Soft background music", "Noisy (cafe, crowded library, etc.)"]

# Tạo transformers
print("Đang tạo pipeline xử lý dữ liệu...")
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
        social_media_usage,
        gender,
        use_technology,
        focus,
        phone_usage,
    ], handle_unknown='use_encoded_value', unknown_value=-1))
])

multi_nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent", fill_value="")),
    ("encoder", MultiLabelTransformer())
])

single_nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(categories=[
        effective_study_time,
        diet_habits,
        study_env,
    ], handle_unknown="ignore", sparse_output=False))
])

# Tạo preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, num_features),
    ("ord_feature", ord_transformer, ord_features),
    ("multi_nom_feature", multi_nom_transformer, multi_nom_features),
    ("single_nom_feature", single_nom_transformer, single_nom_features),
], remainder='passthrough', verbose=True)

# Tạo pipeline cho mô hình
print("Đang tạo pipeline cho mô hình...")
model = LinearRegression()
reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", BoundedRegressor(model, min_val=0, max_val=10))
])

# Huấn luyện mô hình
print("Đang huấn luyện mô hình...")
try:
    reg.fit(x_train, y_train)
    print("Mô hình đã được huấn luyện thành công!")
except Exception as e:
    print(f"Lỗi khi huấn luyện mô hình: {e}")
    raise

# Dự đoán
print("Đang dự đoán trên tập kiểm thử...")
try:
    y_predict = reg.predict(x_test)
    print("Dự đoán thành công!")
except Exception as e:
    print(f"Lỗi khi dự đoán: {e}")
    raise

# In kết quả so sánh
print("\nSo sánh giữa giá trị dự đoán và giá trị thực:")
for i, (pred, actual) in enumerate(zip(y_predict, y_test)):
    print(f"Mẫu {i + 1}: Dự đoán: {pred:.2f}, Thực tế: {actual}")

# Tính toán và in các chỉ số đánh giá
mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(f"\nĐánh giá mô hình:")
print(f"MAE (Sai số tuyệt đối trung bình): {mae:.4f}")
print(f"MSE (Sai số bình phương trung bình): {mse:.4f}")
print(f"R2 (Hệ số xác định): {r2:.4f}")

# Thử nghiệm với RandomForest
print("\nThử nghiệm với RandomForestRegressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", BoundedRegressor(rf_model, min_val=0, max_val=10))
])

try:
    rf_reg.fit(x_train, y_train)
    rf_y_predict = rf_reg.predict(x_test)

    rf_mae = mean_absolute_error(y_test, rf_y_predict)
    rf_mse = mean_squared_error(y_test, rf_y_predict)
    rf_r2 = r2_score(y_test, rf_y_predict)

    print(f"\nĐánh giá mô hình RandomForest:")
    print(f"MAE (Sai số tuyệt đối trung bình): {rf_mae:.4f}")
    print(f"MSE (Sai số bình phương trung bình): {rf_mse:.4f}")
    print(f"R2 (Hệ số xác định): {rf_r2:.4f}")
except Exception as e:
    print(f"Lỗi khi thử nghiệm với RandomForest: {e}")

print("\nHoàn thành!")