import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
%matplotlib inline
mpl.style.use('ggplot')
car = pd.read_csv('/content/quikr_car (1).csv')
backup = car.copy()
car = car[car['year'].str.isnumeric()]
car['year'] = car['year'].astype(int)
car = car[car['Price'] != 'Ask For Price']
car['Price'] = car['Price'].str.replace(',', '').astype(int)
car['kms_driven'] = car['kms_driven'].str.split().str.get(0).str.replace(',', '')
car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)
car = car[~car['fuel_type'].isna()]
car['name'] = car['name'].str.split().str.slice(start=0, stop=3).str.join(' ')
car = car.reset_index(drop=True)
company_encoder = LabelEncoder()
car['company'] = company_encoder.fit_transform(car['company'])
fuel_type_encoder = LabelEncoder()
car['fuel_type'] = fuel_type_encoder.fit_transform(car['fuel_type'])
X = car[['company', 'fuel_type', 'year', 'kms_driven']]
y = car['Price']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
def predict_car_price():
    print("Available car companies:", company_encoder.classes_)
    company = input("Enter car company: ").strip().capitalize()
    if company not in company_encoder.classes_:
        print(f"Invalid company. Available companies: {company_encoder.classes_}")
        return
    fuel_type = input("Enter fuel type: ").strip().capitalize()
    if fuel_type not in ['Petrol', 'Diesel', 'LPG']:
        print("Invalid fuel type. Choose between 'Petrol', 'Diesel', or 'LPG'.")
        return
    kms_driven = int(input("Enter the number of kilometers driven: "))
    year = int(input("Enter the year of manufacture: "))
    company_encoded = company_encoder.transform([company])[0]
    fuel_type_encoded = fuel_type_encoder.transform([fuel_type])[0]
    user_input_data = pd.DataFrame([[company_encoded, fuel_type_encoded, year, kms_driven]],
                                   columns=['company', 'fuel_type', 'year', 'kms_driven'])
    user_input_data = user_input_data.reindex(columns=['company', 'fuel_type', 'year', 'kms_driven'])
    user_input_data_scaled = scaler.transform(user_input_data)
    predicted_price = model.predict(user_input_data_scaled)
    print(f"The predicted price for your car is: ₹{predicted_price[0]:,.2f}")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, model.predict(X_test), label='Test Data', color='blue')
    plt.scatter(y_test.mean(), predicted_price, color='red', label="User Input Prediction", s=100)
    plt.title('Scatter Plot: Actual vs Predicted Prices')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.show()
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=y)
    plt.title('Box Plot: Distribution of Car Prices')
    plt.show()
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=car['fuel_type'], y=car['Price'])
    plt.title('Violin Plot: Car Prices by Fuel Type')
    plt.show()
predict_car_price()
