from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

iris=load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.3,random_state=42)

model=CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=3,
    verbose=False)

model.fit(X_train,y_train)

prediction=model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,prediction))


