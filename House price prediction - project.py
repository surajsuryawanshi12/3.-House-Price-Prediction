import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# for mse
from sklearn.metrics import mean_squared_error

# uploading the data
df = pd.read_csv("Housing (1).csv", index_col=None, encoding="utf-8")
print(df.columns)


df.sort_values(by="area",  ascending=False, inplace=True)
print(df)
x = df[["price","area"]]


print("Top 10 Highest Prices of Houses according to Area")
print(x.head(10))



X = df.drop(columns=['price']) 
# this x for the predictors

y = df['price']                
# it is for missing price

# Split into Train / Test 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print("Train set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])
print("\nTrain Features:\n", X_train)
print("\nTrain Labels:\n", y_train)



# step 4 

a = df.drop(columns=['price'])  # predictors
b = df['price']                 # target variable

# ---- Preprocessing ----
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                    'airconditioning', 'prefarea', 'furnishingstatus']
numeric_cols = [col for col in a.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)

# Build Pipeline 
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train/Test Split 
a_train, a_test, b_train, b_test = train_test_split(
    a, b, test_size=0.25, random_state=42
)

# Train  model 
model.fit(a_train, b_train)

# Evaluate 
train_score = model.score(a_train, b_train)
test_score = model.score(a_test, b_test)

print("Training R²:", train_score)
print("Testing R²:", test_score)

# predications 
b_pred = model.predict(a_test)
print("\nPredicted Prices:", b_pred)



print("Actual Prices:", b_test.values)

# Evaluate predictions using metrics like Mean Squared Error (MSE).
  
mse = mean_squared_error(b_test, b_pred)
print("Mean Squared Error (MSE):", mse)