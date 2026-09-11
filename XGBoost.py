from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

iris=load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.3,random_state=42)

model=XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss')

model.fit(X_train,y_train)

prediction=model.predict(X_test)

print("Actual:",y_test)
print("Predicted:",prediction)
print("Accuracy:",accuracy_score(y_test,prediction))


