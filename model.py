import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("train.csv")

# Select important features
data = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'GarageCars', 'YearBuilt', 'SalePrice']]

# Remove missing values
data = data.dropna()

# Features and target
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'GarageCars', 'YearBuilt']]
y = data['SalePrice']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict new house
new_house = pd.DataFrame([[2000, 3, 2]], columns=X.columns)
price = model.predict(new_house)

print("\nPredicted price:", price[0])

# -----------------------------
# Correlation Heatmap
# -----------------------------
plt.figure(figsize=(6,4))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix Heatmap")
plt.show()

# -----------------------------
# Scatter Plot + Regression Line
# -----------------------------
plt.figure(figsize=(6,4))
sns.regplot(x='GrLivArea', y='SalePrice', data=data)
plt.title("Linear Regression: Area vs Price")
plt.show()