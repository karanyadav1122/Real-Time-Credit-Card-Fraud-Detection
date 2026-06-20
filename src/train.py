from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/raw/creditcard.csv")
MODEL_DIR= Path("models")
MODEL_DIR.mkdir(exist_ok= True)

def main():
  
  df = pd.read_csv(DATA_PATH)
  
  X = df.drop("Class", axis= 1)
  y = df['Class']
  
  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.20, random_state= 42, stratify=y)
  
  mlflow.set_experiment("credit-card-fraud-detection")
  
  with mlflow.start_run():
    model = RandomForestClassifier(
      n_estimators= 100,
      max_depth= 10, 
      random_state= 42,
      n_jobs= -1,
      class_weight= "balanced"
    )

    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    mlflow.log_param("model","RandomForestClassifier")
    mlflow.log_param("n_estimators",100)
    mlflow.log_param("max_depth",10)
    mlflow.log_param("class_weight","balanced")
    
    mlflow.log_metric('precision',precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score",f1)
    mlflow.log_metric("roc_auc",roc_auc)
    
    
    model_path = MODEL_DIR /"fraud_model.pkl"
    joblib.dump(model, model_path)
    
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact(str(model_path))
    
    print("trainig completed")
    print(f"Precision: {precision: .4f}")
    print(f"Recall:{recall: .4f}")
    print(f"F1 score:{f1: .4f}")
    print(f"ROC AUC:{roc_auc: .4f}")
    

if __name__ == "__main__":
  main()    
      