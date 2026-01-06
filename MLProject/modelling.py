import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model():
    mlflow.sklearn.autolog()
    
    # Membaca data dari folder yang sama
    df = pd.read_csv('train_preprocessing.csv')
    
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)

if __name__ == "__main__":
    train_model()