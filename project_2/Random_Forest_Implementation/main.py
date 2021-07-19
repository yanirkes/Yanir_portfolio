import RandomForest as rnf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

#

if __name__ == '__main__':

    df = pd.read_csv('heart.csv')
    x = df.iloc[:, 0:-1]
    x = x.values
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(x, y)

    """Comperison between Sklearn RNF and my RNF"""
    for i in range(20):
        rn = rnf.RandomForest(max_features=8, forest_size=100, max_depth=20)
        rn.fit(X_train, y_train.values.reshape(-1, 1))
        y_hat = rn.predict(X_test)
        f1 = f1_score(y_test, y_hat)
        print(f1)

        skRnf = RandomForestClassifier(n_estimators=100, max_depth=20, max_features=8)
        skRnf.fit(X_train, y_train)
        y_hat_sk = skRnf.predict(X_test)
        print("difference", i, " :", f1 - f1_score(y_test, y_hat_sk))
