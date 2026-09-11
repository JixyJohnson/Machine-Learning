from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples=100,n_features=1,noise=15,random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

prediction = model.predict(X_test)

print("MSE =",mean_squared_error(y_test,prediction))
