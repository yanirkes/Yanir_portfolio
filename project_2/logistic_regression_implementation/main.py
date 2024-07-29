from src import LogisticRegression as lg
from sklearn.linear_model import LogisticRegression as lg
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("sample_data.csv")


    # split the data
    X_train, X_test, y_train, y_test = train_test_split(df[["height", "age"]], df.weight_old, test_size=0.33, random_state=42)

    # train logistic regression implementation
    log = lg.LogisticRegression(alpha=0.1)
    log.fit(X_train, y_train)

    # Use Sklearn to match the methods:
    sk_lg_model = lg()
    sk_lg_model.fit(X_train, y_train)

    # Match results
    # predict
    y_pred_sk = sk_lg_model.predict(X_test)
    y_pred_mymodel = log.predict(X_test)

    # dislay results
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(y_test)), y_pred_sk, label="y_test_sk", color='r')
    plt.scatter(range(len(y_test)), y_pred_mymodel, label="y_test_mymodel", color='b', alpha=0.3)
    plt.show()

    # compare performance
    print(
        f"score of sk model {np.mean(y_test.values == y_pred_sk)}\nscore of mymodel model {np.mean(y_test.values == y_pred_mymodel)}")
    print(f"score of sk model {np.mean(y_pred_mymodel == y_pred_sk)}")