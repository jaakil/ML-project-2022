# First, we must import required items, such as pandas and numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

InflationRawData = pd.read_csv('inflaatio.csv')
print(InflationRawData.shape)
InflationRawData.head()

# Let's change the names to something more reasonable.
InflationRawData.columns = ['Time', 'Rate']
# Turns out, this makes it a lot more reasonable
# Also, let's rename the data. Makes it easier to write.

data = InflationRawData
data.head()

X = np.arange(505).reshape(-1,1)
y = data['Rate'].to_numpy()

print(X.shape)
print(y.shape)
print(X[0])
print(y[0])

# Let's fit a linear regression model
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)

regr = LinearRegression()
regr.fit(X_train, y_train)
y_train_pred = regr.predict(X_train)
tr_error=mean_squared_error(y_train, y_train_pred)
print("The training error is ", tr_error)

y_val_pred = regr.predict(X_val)
val_error = mean_squared_error(y_val, y_val_pred)
print("The validation error is ", val_error)

# Let's visualize the data
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='b', s=8, label='datapoints from the dataframe "inflation"')
plt.xlabel('Time, as an index', size=14)
plt.ylabel('Rate of inflation (%)', size=14)
plt.show()

# Add the linear regression line to it
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='b', s=8, label='datapoints from the dataframe "inflation"')
#plot the linear regression method as l(x)
plt.plot(X_val, y_val_pred, color='r', label='l(x)') 
plt.xlabel('Time, as an index', size=14)
plt.ylabel('Rate of inflation (%)', size=14)
plt.title('Linear regression model', size=15)
plt.legend(loc='best',fontsize=12)

plt.show()

endOfYear =np.array((505,506,507,508,509,510,511,512,513,514,515)).reshape(-1,1)
print(regr.predict(endOfYear))

degrees = [3,4,5,6,7,8,9,10,11,12,13,14,15]
tr_errors = []
val_errors = []

for i in range(len(degrees)):
    
    lin_regr = LinearRegression(fit_intercept=False)
    poly = PolynomialFeatures(degree = degrees[i])
    X_train_poly = poly.fit_transform(X_train)
    lin_regr.fit(X_train_poly, y_train)
    
    y_pred_train = lin_regr.predict(X_train_poly)
    tr_error = mean_squared_error(y_train, y_pred_train)
    
    X_val_poly = poly.fit_transform(X_val)
    y_pred_val = lin_regr.predict(X_val_poly)
    val_error = mean_squared_error(y_val, y_pred_val)
    tr_errors.append(tr_error)
    val_errors.append(val_error)
    
    plt.tight_layout()
    plt.plot(X, lin_regr.predict(poly.transform(X.reshape(-1, 1))), label="Model")    # plot the polynomial regression model
    plt.scatter(X_train, y_train, color="b", s=10, label="Train Datapoints")    # plot a scatter plot of y(maxtmp) vs. X(mintmp) with color 'blue' and size '10'
    plt.scatter(X_val, y_val, color="r", s=10, label="Validation Datapoints")    # do the same for validation data with color 'red'
    plt.xlabel('mintmp')    # set the label for the x/y-axis
    plt.ylabel('maxtmp')
    plt.legend(loc="best")    # set the location of the legend
    plt.title(f'Polynomial degree = {degrees[i]}\nTraining error = {tr_error:.5}\nValidation error = {val_error:.5}')    # set the title
    plt.show()    # show the plot

print(tr_errors)
print(val_errors)

lin_regr=LinearRegression(fit_intercept=False)
poly = PolynomialFeatures(degree=5)
X_train_poly = poly.fit_transform(X_train)
lin_regr.fit(X_train_poly, y_train)
endOfYear_transformed = poly.fit_transform(endOfYear)

print(lin_regr.predict(endOfYear_transformed))


lin_regr=LinearRegression(fit_intercept=False)
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
lin_regr.fit(X_train_poly, y_train)
endOfYear_transformed = poly.fit_transform(endOfYear)

print(lin_regr.predict(endOfYear_transformed))
