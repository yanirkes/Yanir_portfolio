import numpy as np
import pandas as pd

class LogisticRegression:

    def __init__(self, link_function='logit', alpha=0.001, learning_rate=0.001, max_iter=50):
        """
        Initializes the LogisticRegression model with given parameters.

        Args:
            link_function (str): The link function to use ('logit' by default).
            alpha (float): Convergence threshold.
            learning_rate (float): The learning rate for the Newton-Raphson update.
            max_iter (int): Maximum number of iterations for convergence.
        """
        self.iter = 0
        self.link_function = link_function
        self.columns_names = []
        self.loss = None
        self.coefficients = np.empty(0)
        self.intercept = np.array([1])
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit(self, X, y):
        """
        Fits the logistic regression model to the training data.

        Args:
            X (array-like): The input features.
            y (array-like): The target variable.
        """
        # Verify input values
        self.verify_x(X)
        n = X.shape[0]
        X = np.c_[np.ones(len(X)), X]  # Add intercept term
        self.verify_y(y)
        y = np.asarray(y).reshape(-1, 1)

        self.newton_rabpson(X, y, n)

    def cost_function(self, X, y, n, beta, epsilon = 1e-9):
        """
        Calculates the cost function for logistic regression.

        Args:
            X (array-like): The input features.
            y (array-like): The target variable.
            n (int): Number of samples.
            beta (array-like): Coefficients.
            epsilon (float): used to avoid log on 0

        Returns:
            float: The cost function value.
        """
        sig_res = self.sigmoid(X.dot(beta))
        first_term = np.matmul(y.T, np.log(sig_res + epsilon))
        second_term = (1 - y).T.dot(np.log(1 - sig_res + epsilon))
        cost = -(first_term + second_term) / n
        return cost

    @staticmethod
    def sigmoid(rho):
        """
        Computes the sigmoid function.

        Args:
            rho (array-like): Linear combination of inputs and weights.

        Returns:
            array-like: Sigmoid of rho.
        """
        return 1 / (1 + np.exp(-rho))

    def gradient(self, X, y, beta):
        """
        Computes the gradient of the cost function.

        Args:
            X (array-like): The input features.
            y (array-like): The target variable.
            beta (array-like): Coefficients.

        Returns:
            array-like: The gradient vector.
        """
        sigma = self.sigmoid(X.dot(beta))
        return (sigma - y).T.dot(X)

    def hessian(self, X, n, beta):
        """
        Computes the Hessian matrix of the cost function.

        Args:
            X (array-like): The input features.
            n (int): Number of samples.
            beta (array-like): Coefficients.

        Returns:
            array-like: The Hessian matrix.
        """
        h = self.sigmoid(X.dot(beta))
        w = h * (1 - h)
        diag_w = np.diag(w.ravel())
        return (1 / n) * X.T.dot(diag_w) @ X

    @staticmethod
    def verify_x(X):
        """
        Verifies the input features X.

        Args:
            X (array-like): The input features.
        """
        try:
            if not isinstance(X, np.ndarray):
                X = LogisticRegression.convert_to_np(X)
            if X.size == 0:
                raise ValueError("X shouldn't be empty")
            if X.shape[0] / X.shape[1] < 5:
                raise ValueError("There are too many features per row in the data")
        except ValueError as e:
            print("Validation error:", e)
        except Exception as e:
            print("Unexpected error:", e)

    @staticmethod
    def verify_pred_x(X):
        """
        Verifies the input features X for prediction.

        Args:
            X (array-like): The input features.
        """
        try:
            if not isinstance(X, np.ndarray):
                X = LogisticRegression.convert_to_np(X)
            if X.size == 0:
                raise ValueError("X shouldn't be empty")
        except ValueError as e:
            print("Validation error:", e)
        except Exception as e:
            print("Unexpected error:", e)

    @staticmethod
    def verify_y(y):
        """
        Verifies the target variable y.

        Args:
            y (array-like): The target variable.
        """
        try:
            if y.size == 0:
                raise ValueError("y shouldn't be empty")
            if np.isnan(y).any():
                raise ValueError("There are NaNs in the y target")
            if np.unique(y).size > 2:
                raise ValueError("There are too many unique values in y")
            if not np.all(np.isin(y, [0, 1])):
                raise ValueError("y should contain only boolean values, 0 and 1")
            if y.shape[0] < 5:
                raise ValueError("There are too few instances for y")
        except ValueError as e:
            print("Validation error:", e)
        except Exception as e:
            print("Unexpected error:", e)

    def newton_rabpson(self, X, y, n):
        """
        Fits the model using the Newton-Raphson method.

        Args:
            X (array-like): The input features.
            y (array-like): The target variable.
            n (int): Number of samples.
        """
        beta = np.ones((X.shape[1], 1)) * 0.002
        loss = self.cost_function(X, y, n, beta)
        g = 1
        self.iter = 0

        while True:
            if np.linalg.norm(g) < self.alpha or self.iter > self.max_iter:
                break
            g = self.gradient(X, y, beta)
            H = self.hessian(X, n, beta)
            beta = beta - self.learning_rate * np.linalg.inv(H).dot(g.T)
            loss = loss - self.cost_function(X, y, n, beta)
            self.iter += 1
        self.coefficients = beta[1:,:]
        self.intercept = beta[0,:]
        self.loss = loss

    def fit_logit(self):
        """
        Placeholder for fitting using the logit link function.
        """
        raise "The methods hasn't been implemented yet"

    def fit_loglog(self):
        """
        Placeholder for fitting using the log-log link function.
        """
        raise "The methods hasn't been implemented yet"

    def predict(self, X, threshold=0.5):
        """
        Predicts the class labels for the input data.

        Args:
            X (array-like): The input features.
            threshold (float): The threshold for classifying probabilities.

        Returns:
            array-like: Predicted class labels.
        """
        self.verify_pred_x(X)

        p_hat = self.predict_proba(X, False)
        return [1 if i > threshold else 0 for i in p_hat]
        # return np.where(p_hat >= threshold, 1, 0)

    def predict_proba(self, X, add_intercept=True):
        """
        Predicts the class probabilities for the input data.

        Args:
            X (array-like): The input features.
            add_intercept (bool): Whether to add an intercept term.

        Returns:
            array-like: Predicted class probabilities.
        """
        # self.verify_pred_x(X)
        # rho = X.dot(self.coefficients) + self.intercept[0]

        weight_ = self.coefficients
        inter_ = self.intercept[0]
        rho = np.dot(X, weight_) + inter_
        p = self.sigmoid(rho)
        return p


    @staticmethod
    def score(y, y_hat):
        """
        Computes the accuracy score.

        Args:
            y (array-like): True class labels.
            y_hat (array-like): Predicted class labels.

        Returns:
            float: Accuracy score.
        """
        if not isinstance(y, np.ndarray):
            raise ValueError("The y type is not a numpy array")
        return np.mean(y == y_hat)

    @staticmethod
    def convert_to_np(x):
        """
        Converts input data to a numpy array.

        Args:
            x (array-like): The input data.

        Returns:
            array-like: Converted numpy array.
        """
        if isinstance(x, pd.DataFrame):
            return x.to_numpy()
        elif isinstance(x, pd.Series):
            return x.to_numpy()
        elif isinstance(x, list):
            return np.array(x)
        else:
            return x
