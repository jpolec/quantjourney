"""
  Volatility analysis

  author: jpolec@gmail.com
  date: 16-01-2024
"""

import yfinance as yf
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
import numpy as np
from arch import arch_model
from arch.__future__ import reindexing

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot

stocks = 'AAPL'
start = datetime.datetime(2012,1,1)
end = datetime.datetime(2023,1,1)
dax = yf.download(stocks, start, end, interval='1d')

ret = 100 * (dax.pct_change()[1:]["Adj Close"])
realized_vol = ret.rolling(5).std()

# plt.figure(figsize=(12, 8))
# plt.plot(realized_vol)
# plt.show()

# --------------------------------------------------
print("ARCH, p=1")
n = (datetime.datetime.strptime('2023/1/1', "%Y/%m/%d") - datetime.datetime.strptime('2020/1/1', "%Y/%m/%d")).days
split_date = ret.iloc[-n:].index
arch = arch_model(ret, mean='zero', vol='ARCH', p=1).fit(disp='off')
print(arch.summary())

# Extract the conditional volatility (standard deviation)
conditional_volatility = arch.conditional_volatility

# Plot the actual returns and the conditional volatility
# plt.figure(figsize=(12, 8))
# plt.plot(ret.index, ret, label='Actual Returns')
# plt.plot(conditional_volatility.index, conditional_volatility, label='Conditional Volatility', linestyle='--')

# plt.title('Actual Returns vs. Conditional Volatility from ARCH(1)')
# plt.xlabel('Date')
# plt.ylabel('Returns / Volatility')
# plt.legend()
# plt.show()

# --------------------------------------------------
print("ARCH, p=var")
bic_arch = []
for p in range(1, 5):
    arch = arch_model(ret, mean='zero', vol='ARCH', p=p).fit(disp='off') 
    bic_arch.append(arch.bic)
    if arch.bic == np.min(bic_arch): 
            best_param = p
arch = arch_model(ret, mean='zero', vol='ARCH', p=best_param).fit(disp='off') 
print(arch.summary())
forecast = arch.forecast(start = split_date[0])
forecast_arch = forecast
rmse_arch = np.sqrt(mse(realized_vol[-n:]/100,
                        np.sqrt(forecast_arch.variance.iloc[-len(split_date):]/100)))
print("The RMSE value of ARCH model is {:.4f}".format(rmse_arch))

# plt.figure(figsize=(12, 6))
# plt.plot(realized_vol,label='Actual Volatility')
# # Extracting the forecasted volatility from the model's forecast
# predicted_volatility = np.sqrt(forecast_arch.variance.iloc[-len(split_date):])
# # Plotting the predicted volatility
# plt.plot(predicted_volatility, label='ARCH Predicted Volatility', color='red', linestyle='-')
# plt.title('Actual vs. ARCH Predicted Volatility')
# plt.xlabel('Date')
# plt.ylabel('Volatility')
# plt.legend()
# plt.show()

# --------------------------------------------------
print("GARCH")
garch = arch_model(ret, mean='zero', vol='GARCH', p=1, o=0, q=1).fit(disp='off')
print(garch.summary())

bic_garch = []
for p in range(1, 5):
 for q in range(1, 5):
     garch = arch_model(ret, mean='zero',vol='GARCH', p=p, o=0, q=q).fit(disp='off')
     bic_garch.append(garch.bic)
     if garch.bic == np.min(bic_garch):
         best_param = p, q
garch = arch_model(ret, mean='zero', vol='GARCH',
                p=best_param[0], o=0, q=best_param[1]).fit(disp='off')
print(garch.summary())

forecast = garch.forecast(start=split_date[0])
forecast_garch = forecast
rmse_garch = np.sqrt(mse(realized_vol[-n:] / 100,
                         np.sqrt(forecast_garch \
                                 .variance.iloc[-len(split_date):] / 100)))
print('The RMSE value of GARCH model is {:.6f}'.format(rmse_garch))

# plt.figure(figsize=(12, 6))
# plt.plot(realized_vol,label='Actual Volatility')
# # Extracting the forecasted volatility from the model's forecast
# predicted_volatility = np.sqrt(forecast_garch.variance.iloc[-len(split_date):])
# # Plotting the predicted volatility
# plt.plot(predicted_volatility, label='GARCH Predicted Volatility', color='red', linestyle='-')
# plt.title('Actual vs. GARCH Predicted Volatility')
# plt.xlabel('Date')
# plt.ylabel('Volatility')
# plt.legend()
# plt.show()

# --------------------------------------------------
q=2
print("GJR GARCH")
bic_gjr_garch = []
for p in range(1,5):
    gjrgarch = arch_model(ret, mean='zero', p = p, o=1, q=q).fit(disp='off')
    bic_gjr_garch.append(gjrgarch.bic)
    if gjrgarch.bic == np.min(bic_gjr_garch):
        best_param = p,q
gjrgarch = arch_model(ret, mean='zero', p=best_param[0], q=best_param[1], o=1).fit(disp='off')
print(gjrgarch.summary())
forecast = gjrgarch.forecast(start=split_date[0])
forecast_gjrgarch = forecast

rmse_gjr_garch = np.sqrt(mse(realized_vol[-n:]/100,
                             np.sqrt(forecast_garch.variance.iloc[-len(split_date):]/100)))
print("The RMSE value of GJR GARCH model is {:.6f}".format(rmse_gjr_garch))

# plt.figure(figsize=(12, 6))
# plt.plot(realized_vol,label='Actual Volatility')
# # Extracting the forecasted volatility from the model's forecast
# predicted_volatility = np.sqrt(forecast_gjrgarch.variance.iloc[-len(split_date):])
# # Plotting the predicted volatility
# plt.plot(predicted_volatility, label='GARCH GJR Predicted Volatility', color='red', linestyle='-')
# plt.title('Actual vs. GARCH GJR Predicted Volatility')
# plt.xlabel('Date')
# plt.ylabel('Volatility')
# plt.legend()
# plt.show()

# --------------------------------------------------
bic_egarch = []
print("EGARCH")
for p in range(1,5):
    for q in range(1,5):
        egarch = arch_model(ret, mean='zero', vol='EGARCH',
                            p=p, q=q).fit(disp='off')
        bic_egarch.append(egarch.bic)
        if egarch.bic == np.min(bic_egarch):
            best_param = p,q
egarch = arch_model(ret, mean='zero', vol='EGARCH',
                    p=best_param[0], q=best_param[1]).fit(disp='off')
print(egarch.summary())

forecast = egarch.forecast(start=split_date[0])
forecast_egarch = forecast
rmse_egarch = np.sqrt(mse(realized_vol[-n:]/100,
                          np.sqrt(forecast_egarch.variance.iloc[-len(split_date):]/100)))
print("The RMSE value of EFARCH model is {:.6f}".format(rmse_egarch))

# plt.figure(figsize=(12, 6))
# plt.plot(realized_vol,label='Actual Volatility')
# # Extracting the forecasted volatility from the model's forecast
# predicted_volatility = np.sqrt(forecast_egarch.variance.iloc[-len(split_date):])
# # Plotting the predicted volatility
# plt.plot(predicted_volatility, label='EGARCH Predicted Volatility', color='red', linestyle='-')
# plt.title('Actual vs. EGARCH Predicted Volatility')
# plt.xlabel('Date')
# plt.ylabel('Volatility')
# plt.legend()
# plt.show()

# --------------------------------------------------
from sklearn.svm import SVR
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV

realized_vol = ret.rolling(5).std()
realized_vol = pd.DataFrame(realized_vol)
realized_vol.reset_index(drop=True, inplace=True)
retruns_svm = ret ** 2
returns_svm = retruns_svm.reset_index()

del returns_svm["Date"]

X = pd.concat([realized_vol, returns_svm], axis = 1, ignore_index=True)

X = X[4:].copy()
X = X.reset_index()
X.drop("index", axis = 1, inplace = True)

print("SVR with Linear Kernel")
realized_vol = realized_vol.dropna().reset_index()
realized_vol.drop('index', axis=1, inplace=True)
svr_poly = SVR(kernel='poly',degree=2)
svr_lin  = SVR(kernel='linear')
svr_rbf  = SVR(kernel='rbf')

para_grid = {'gamma':sp_rand(),
             'C': sp_rand(),
             'epsilon': sp_rand()}
clf =RandomizedSearchCV(svr_lin, para_grid)
clf.fit(X.iloc[:-n].values,
        realized_vol.iloc[1:-(n-1)].values.reshape(-1,))
predict_svr_lin = clf.predict(X.iloc[-n:])
predict_svr_lin = pd.DataFrame(predict_svr_lin)
predict_svr_lin.index = ret.iloc[-n:].index
rmse_svr =np.sqrt(mse(realized_vol.iloc[-n:]/100,
                      predict_svr_lin/100))
print("The RMSE value of SVR with Linear Kernel is {:.4f}".format(rmse_svr))

plt.figure(figsize=(12, 6))
plt.plot(realized_vol,label='Actual Volatility')
# Extracting the forecasted volatility from the model's forecast
plt.plot(predict_svr_lin, label='SVR Predicted Volatility', color='red', linestyle='-')
plt.title('Actual vs. SVR Predicted Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.show()

# --------------------------------------------------
print("SVR with RBF Kernel")
para_grid ={'gamma': sp_rand(),
         'C': sp_rand(),
         'epsilon': sp_rand()}
clf = RandomizedSearchCV(svr_rbf, para_grid)
clf.fit(X.iloc[:-n].values,
     realized_vol.iloc[1:-(n-1)].values.reshape(-1,))
predict_svr_rbf = clf.predict(X.iloc[-n:])
predict_svr_rbf = pd.DataFrame(predict_svr_rbf)
predict_svr_rbf.index = ret.iloc[-n:].index
rmse_svr_rbf = np.sqrt(mse(realized_vol.iloc[-n:] / 100,
                                    predict_svr_rbf / 100))
print('The RMSE value of SVR with RBF Kernel is  {:.6f}'
   .format(rmse_svr_rbf))


print("SVT with Polynomial Kernel")
para_grid = {'gamma': sp_rand(),
                     'C': sp_rand(),
                     'epsilon': sp_rand()}
clf = RandomizedSearchCV(svr_poly, para_grid)
clf.fit(X.iloc[:-n].values,
                 realized_vol.iloc[1:-(n-1)].values.reshape(-1,))
predict_svr_poly = clf.predict(X.iloc[-n:])
predict_svr_poly = pd.DataFrame(predict_svr_poly)
predict_svr_poly.index = ret.iloc[-n:].index
rmse_svr_poly = np.sqrt(mse(realized_vol.iloc[-n:] / 100,
                                     predict_svr_poly / 100))
print('The RMSE value of SVR with Polynomial Kernel is {:.6f}'\
               .format(rmse_svr_poly))


# --------------------------------------------------
from sklearn.neural_network import MLPRegressor
print("NN")
# Configuring the NN model with three hidden layers and varying neuron numbers
NN_vol = MLPRegressor(learning_rate_init=0.001, random_state=1)
para_grid_NN = {'hidden_layer_sizes': [(100,50), (50,50), (10,100)],
                'max_iter':  [250, 500, 1000, 1250, 1500],
                'alpha':[0.00005, 0.0005, 0.005]}

clf = RandomizedSearchCV(NN_vol, para_grid_NN)
clf.fit(X.iloc[:-n].values,
        realized_vol.iloc[1:-(n-1)].values.reshape(-1,))

NN_predictions = clf.predict(X.iloc[-n:])
NN_predictions = pd.DataFrame(NN_predictions)
NN_predictions.index = ret.iloc[-n:].index

rmse_NN = np.sqrt(mse(realized_vol.iloc[-n:]/100,
                      NN_predictions/100))
print("The RMSE value of NN is {:.6f}".format(rmse_NN))


from sklearn.neural_network import MLPRegressor

print("NN")
# Configuring the NN model with three hidden layers and varying neuron numbers
NN_vol = MLPRegressor(learning_rate_init=0.001, random_state=1)
para_grid_NN = {'hidden_layer_sizes': [(100,50), (50,50), (10,100)],
                'max_iter':  [250, 500, 1000, 1250, 1500],
                'alpha':[0.00005, 0.0005, 0.005]}

clf = RandomizedSearchCV(NN_vol, para_grid_NN)
clf.fit(X.iloc[:-n].values,
        realized_vol.iloc[1:-(n-1)].values.reshape(-1,))

NN_predictions = clf.predict(X.iloc[-n:])
NN_predictions = pd.DataFrame(NN_predictions)
NN_predictions.index = ret.iloc[-n:].index

rmse_NN = np.sqrt(mse(realized_vol.iloc[-n:]/100,
                      NN_predictions/100))
print("The RMSE value of NN is {:.6f}".format(rmse_NN))

# --------------------------------------------------
print("DL")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential(
    [layers.Dense(256, activation = "relu"),
     layers.Dense(256, activation = "relu"),
     layers.Dense(128, activation = "relu"),
     layers.Dense(1, activation = "linear")])

model.compile(loss= 'mse', optimizer = "rmsprop")

epochs_trial = np.arange(100,400,4)
batch_trial  = np.arange(100,400,4)
DL_pred = []
DL_RMSE = []

for i, j, k in zip(range(4), epochs_trial, batch_trial):
    model.fit(X.iloc[:-n].values,
              realized_vol.iloc[1:-(n-1)].values.reshape(-1),
              batch_size=k, epochs = j, verbose =False)
    DL_predict = model.predict(np.asarray(X.iloc[-n:]))
    DL_RMSE.append(np.sqrt(mse(realized_vol.iloc[-n:]/100,
                               DL_predict.flatten()/100)))

    DL_pred.append(DL_predict)
    print("DL_RMSE_{}:{:.6f}".format(i+1, DL_RMSE[i]))

DL_predict = pd.DataFrame(DL_pred[DL_RMSE.index(min(DL_RMSE))])
DL_predict.index = ret.iloc[-n:].index

rmse_DL = np.sqrt(mse(realized_vol.iloc[-n:]/100,
                      DL_predict/100))
print("The RMSE value of NN is {:.6f}".format(rmse_DL))

# --------------------------------------------------
print("Bayesian GARCH")
import pyflux as pf
from scipy.stats import kurtosis

model = pf.GARCH(ret.values, p=1, q=1)
print(model.latent_variables)

model.adjust_prior(1, pf.Normal())
model.adjust_prior(2, pf.Normal())

x = model.fit(method='M-H', iterations='10000')
print(x.summary())

bayesian_prediction = model.predict_is(n, fit_method='M-H')
bayesian_RMSE = np.sqrt(mse(realized_vol.iloc[-n:] / 100,
                                  bayesian_prediction.values / 100))
print('The RMSE of Bayesian model is {:.6f}'.format(bayesian_RMSE))
