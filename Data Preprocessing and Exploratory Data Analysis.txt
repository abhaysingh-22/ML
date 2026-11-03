import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(URL)
print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum())

# 1. Data cleaning
df = df.drop_duplicates()
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])  # irrelevant / ~77% missing

# 2. Missing value handling
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
print("\nMissing after handling:", df.isnull().sum().sum())

# 3. Statistical analysis
print("\n", df.describe())
print("\nCorrelation with Survived:\n", df.corr(numeric_only=True)["Survived"])

# 4. Visualization (histograms and boxplots)
num = ["Age", "Fare", "SibSp", "Parch"]
df[num].hist(bins=20, figsize=(10, 6))
plt.tight_layout()
plt.show()
df[num].plot(kind="box", subplots=True, layout=(1, 4), figsize=(12, 4))
plt.tight_layout()
plt.show()

# 5. Encoding
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})            # label encoding
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)  # one-hot encoding

# 6. Train-test split (before scaling to avoid data leakage)
X, y = df.drop(columns="Survived"), df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Feature scaling (fit on train only)
cols = ["Age", "Fare"]
scaler = StandardScaler()
X_train, X_test = X_train.copy(), X_test.copy()
X_train[cols] = scaler.fit_transform(X_train[cols])
X_test[cols] = scaler.transform(X_test[cols])

print("\nTrain:", X_train.shape, "Test:", X_test.shape)
print(X_train.head())