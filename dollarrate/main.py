"""
Programm for predicting dollar course from 2 datasets: us dollar prices and oil prices at the same time for severl years

I used few prediction models to figure out the best pridiction model.
"""
import pandas as pd
import numpy as np

usd_rate = pd.read_excel("/content/RC_F01_12_2017_T01_12_2020.xlsx")

brent_oil = pd.read_excel("/content/RBRTEd.xls",sheet_name=1, names=['date', 'oil_price'], skiprows=2)

usd_rate.curs.plot()

brent_oil.oil_price.plot()


df = usd_rate.set_index('data').join(brent_oil.set_index('date')) #объединяем таблицы по дате

df.drop(['nominal', 'cdx'], inplace=True, axis=1) # удаляем ненужные строки

df.fillna(method='ffill', inplace=True) #заменяем пропуски предыдущими значениями

df.reset_index(inplace=True) # переиндескация строк, т.к они могли изменить свой порядок после предыдущих действий

#feature engineering
df['year'] = df['data'].dt.year # из даты берем год
df['month'] = df['data'].dt.month # из даты берем год
df['day'] = df['data'].dt.dayofweek # из даты берем год

past_days = 7
for day in range(past_days):
  n=day+1
  df[f"day_lag_{n}"] = df['curs'].shift(n)
  df[f'oil_lag_{n}'] = df['oil_price'].shift(n)
  df[f"mult_{n}"] = df[f"day_lag_{n}"] * df[f'oil_lag_{n}']

df['usd_mean_week'] = df['curs'].shift(1).rolling(window = 7).median() # шифт(1), чтобы не учитвать завтрашний день, иначе ошибочная модель 
df['oil_mean_week'] = df['oil_price'].shift(1).rolling(window = 7).median()

final_df = pd.get_dummies(df, columns=['year', 'month', 'day']).drop(['data','oil_price'], axis=1)[7:]

X = final_df.drop('curs', axis=1) # данные на основе которых делаем прогноз
y = final_df.curs # то, что мы хотим спрогнозировать

# делим датасеты на train set и test set
X_train = X[:700]
y_train = y[:700]
X_test = X[700:]
y_test = y[700:]

from sklearn.linear_model import LinearRegression

LinReg = LinearRegression()
LinReg.fit(X_train, y_train)

prediction = LinReg.predict(X_test)

from sklearn.metrics import mean_absolute_error

print("MAE = ", mean_absolute_error(y_test, prediction))

# ошибка = 0, чето тут не так. Проверим коэфы

pd.DataFrame(data=LinReg.coef_, index=X.columns, columns=['value']).sort_values(by='value', ascending = False)
# usd_mean_week большой, он уже имеет курс на сегодня. Нельзя его юзать в таком виде

# пробуем другую модель
from sklearn.linear_model import Ridge, Lasso

ridge_regression = Ridge()
lasso_regression = Lasso()
ridge_regression.fit(X_train, y_train)
lasso_regression.fit(X_train, y_train)

ridge_prediction = ridge_regression.predict(X_test)
lasso_prediction = lasso_regression.predict(X_test)

def mae(labels, predictions):
  return mean_absolute_error(labels, predictions)

lasso_mae = mae(y_test, lasso_prediction)
ridge_mae = mae(y_test, ridge_prediction)

print("lasso mae = ", lasso_mae)
print("ridge mae = ", ridge_mae)

pd.DataFrame(data=lasso_regression.coef_, index=X.columns, columns=['value']).sort_values(by='value', ascending = False)

# попробуем сделать регрессию через дерево решений
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_prediction = rf.predict(X_test)
rf_mae = mae(y_test, rf_prediction)

print("rf mae = ", rf_mae)
