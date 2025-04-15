
# 🎓 GPA Prediction using Student Study Habits

This project uses data about students' daily study habits to predict their latest semester GPA using a Linear Regression model and a comprehensive preprocessing pipeline.

## 📁 Project Structure

```
├── model_final.py                 # Main script for training and testing the model
├── gpa-collections-adjusted.csv   # Input dataset
├── requirement.txt                # List of required Python packages
```

---

## ⚙️ Setup Instructions

### 🔹 1. Clone this repository

```bash
git clone https://github.com/nhu220840/predict-gpa-from-study-habits.git
cd gpa-prediction
```

---

### 🔹 2. Create a Virtual Environment

#### 🪟 Windows (Command Prompt)

```cmd
python -m venv venv
venv\Scripts\activate
```

#### 🍎 macOS / 🐧 Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 🔹 3. Install Required Packages

```bash
pip install -r requirement.txt
```

---

## ▶️ Run the Project

```bash
python model_final.py
```

Once run, the script will:
- Preprocess the data based on different feature types (numerical, ordinal, multi-label, nominal).
- Apply a Linear Regression model bounded between 0 and 10 GPA.
- Evaluate the model using MAE, MSE, and R² score.

---

## 📊 Output Example

```
Predicted value: 7.25. Actual value: 7.00
Predicted value: 6.85. Actual value: 7.20

Evaluate model:
MAE: 0.42
MSE: 0.31
R2: 0.81
```

---

## 📌 Notes

- The dataset must be named exactly `gpa-collections-adjusted.csv`.
- If you want to generate a detailed profiling report using `ydata-profiling`, you can uncomment the line `profile.to_file(...)` in `model_final.py`.

---

## 📬 Contact

If you encounter any issues or need further information, feel free to open an [issue](https://github.com/nhu220840/predict-gpa-from-study-habits.git) or contact directly.

---

## 📝 License

MIT License © Nhu Do Nguyen Gia
