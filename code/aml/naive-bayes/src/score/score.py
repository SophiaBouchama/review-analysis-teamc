
y_pred = clf.predict(X_val_vect)
y_pred_test = clf.predict(X_test_vect)

print(classification_report(y_val, y_pred))
mlflow.log_metric("val accuracy", accuracy_score(y_val, y_pred))
mlflow.log_metric("test accuracy", accuracy_score(y_test, y_pred_test))