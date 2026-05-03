"""
============================================================
  House Price Prediction - ML Project
  Author  : Atikul Islam
  GitHub  : github.com/AtikulBank
  Email   : a01759831040@gmail.com
============================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   HOUSE PRICE PREDICTION — by Atikul Islam")
print("=" * 60)

np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'area_sqft'    : np.random.randint(500, 5000, n),
    'bedrooms'     : np.random.randint(1, 6, n),
    'bathrooms'    : np.random.randint(1, 4, n),
    'age_years'    : np.random.randint(0, 50, n),
    'garage'       : np.random.randint(0, 3, n),
    'location'     : np.random.choice(['Urban', 'Suburban', 'Rural'], n),
    'condition'    : np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n),
})

loc_map  = {'Urban': 1.5, 'Suburban': 1.0, 'Rural': 0.6}
cond_map = {'Excellent': 1.3, 'Good': 1.0, 'Fair': 0.8, 'Poor': 0.6}

data['price'] = (
    data['area_sqft'] * 150
    + data['bedrooms'] * 10000
    + data['bathrooms'] * 8000
    - data['age_years'] * 500
    + data['garage'] * 7000
    + data['location'].map(loc_map) * 50000
    + data['condition'].map(cond_map) * 30000
    + np.random.normal(0, 15000, n)
).astype(int)

print(f"\n📊 Dataset Shape : {data.shape}")
print(f"   Price Range   : ${data['price'].min():,} — ${data['price'].max():,}")
print(f"   Avg Price     : ${data['price'].mean():,.0f}")

le = LabelEncoder()
data['location_enc']  = le.fit_transform(data['location'])
data['condition_enc'] = le.fit_transform(data['condition'])
data['total_rooms']   = data['bedrooms'] + data['bathrooms']

features = ['area_sqft', 'bedrooms', 'bathrooms', 'age_years',
            'garage', 'location_enc', 'condition_enc', 'total_rooms']

X = data[features]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

models = {
    'Linear Regression'  : LinearRegression(),
    'Ridge Regression'   : Ridge(alpha=1.0),
    'Random Forest'      : RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting'  : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

print("\n🤖 Model Results:")
print("-" * 50)
print(f"{'Model':<22} {'RMSE':>10} {'R² Score':>10}")
print("-" * 50)

best_model = None
best_r2    = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    print(f"{name:<22} ${rmse:>9,.0f} {r2:>10.4f}")
    if r2 > best_r2:
        best_r2    = r2
        best_model = (name, model)

print("-" * 50)
print(f"\n🏆 Best Model : {best_model[0]}")
print(f"   R² Score   : {best_r2:.4f}")

new_house = pd.DataFrame([{
    'area_sqft': 2000, 'bedrooms': 3, 'bathrooms': 2,
    'age_years': 10, 'garage': 1, 'location_enc': 2,
    'condition_enc': 1, 'total_rooms': 5
}])
predicted = best_model[1].predict(scaler.transform(new_house))[0]
print(f"\n🏠 Predicted Price : ${predicted:,.0f}")
print("\n✅ Done! — github.com/AtikulBank")
