import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

def train_model():
    base_path = os.path.dirname(__file__)
    csv_path = os.path.join(base_path, 'train_preprocessing.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {csv_path}")

    df = pd.read_csv(csv_path)
    
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_experiment("Titanic_Experiment")

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        
        # Log manual
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        print(f"Model trained with accuracy: {accuracy}")

if __name__ == "__main__":
    train_model()