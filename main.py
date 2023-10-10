from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from src.genetic_algorithm import GeneticAlgorithm
from src.individual import Individual

def main():
    data = pd.read_csv("data/Credit.csv")

    cat_columns = list(data.select_dtypes(include=["object"]).columns)
    le = LabelEncoder()
    for col in cat_columns:
        data[col] = le.fit_transform(data[col])

    X_train, X_test, y_train, y_test = train_test_split(
        data.iloc[:, :-1],
        data.iloc[:, -1],
        test_size=0.3,
        random_state=0
    )

    num_columns = list(set(data.columns) - set(cat_columns))
    scaler = MinMaxScaler()
    X_train[num_columns] = scaler.fit_transform(X_train[num_columns])
    X_test[num_columns] = scaler.transform(X_test[num_columns])

    ga = GeneticAlgorithm(6, X_train, X_test, y_train, y_test)

    result = ga.solve(generations=10)

    print(result)

if __name__ == "__main__":
    main()