import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

def train_model():
    # Mengatur Path agar fleksibel
    base_path = os.path.dirname(__file__)
    csv_path = os.path.join(base_path, 'train_preprocessing.csv')
    
    # Baca Dataset
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_experiment("Titanic_Experiment")

    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        
        # Log secara manual
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("n_estimators", 100)
        
        # Simpan model ke MLflow
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Berhasil! Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    train_model()