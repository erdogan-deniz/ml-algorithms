import numpy
import pandas
import numpy as np
from typing import Union
import pandas as pd

EPS = 1e-15


def sigmoid(x):
    res = numpy.exp(-x)
    tmp = 1 / (1 + res)

    return tmp


def converter(x):
    return (x >= 0.5) * 1


class MyLogReg:
    def __init__(self, n_iter: int = 10, learning_rate: float = 0.1, metric: str = None) -> None:
        """
        The method initializes the fields of the class.

        :param n_iter: A count of gradient iterations.
        :param learning_rate: A multiplier of gradient step.
        """

        self.__metric = metric

        self.__weights: Union[numpy.array, None] = None

        self.__best_score = None

        # Set n_iter value:
        if 5 <= n_iter <= 2500:
            self.__n_iter: int = n_iter
        else:
            raise "Error! Incorrect n_iter argument value."

        # Set learning rate value:
        if 0.00001 <= learning_rate <= 100:
            self.__learning_rate: float = learning_rate
        else:
            raise "Error! Incorrect learning_rate argument value."

    def fit(self, x: pandas.DataFrame, y: pandas.Series, verbose: Union[int, bool] = False) -> None:
        """
        The method trains the model by finding the best weights.

        :param y: A target variable.
        :param verbose: A logging flag.
        :param x: Features for fitting.
        :return: None.
        """

        # Add the bias column:
        x.insert(0, "w0", 1.0)
        self.__weights = numpy.ones(x.shape[1])

        for step in range(1, self.__n_iter + 1):
            # Predict y:
            calculated_y: numpy.array = x @ self.__weights

            # Convert to probability of model confidence:
            converted_y: numpy.array = 1 / (1 + numpy.e ** (-calculated_y))

            # Calculate loss value:
            loss_value: float = -1 * numpy.mean((y * numpy.log(converted_y + EPS) +
                                                 (1 - y) * numpy.log(1 - converted_y + EPS)))

            # Calculate gradient of loss function:
            gradient: numpy.array = (converted_y - y) @ x / len(y)

            # Update model weights:
            self.__weights -= gradient * self.__learning_rate

        self.__best_score = self.calc_metric(self.predict_proba(x), y)
        pass

    # 1-0
    def predict(self, x: pandas.DataFrame):

        res = converter(self.predict_proba(x))

        return res.astype(int)

    # вероятность
    def predict_proba(self, x: pandas.DataFrame):

        if "w0" not in x.columns:
            x.insert(0, "w0", 1.0)

        return sigmoid(numpy.array(x @ self.__weights))


    def calc_metric(self, y_proba, pred_y):
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        y = converter(y_proba)

        for i in range(len(y)):
            if (y[i] == pred_y[i]) and (y[i] == 1):
                tp += 1
            elif (y[i] == pred_y[i]) and (y[i] == 0):
                tn += 1
            elif (y[i] != pred_y[i]) and (y[i] == 0):
                fp += 1
            else:
                fn += 1

        if self.__metric == "accuracy":
            return (tp + tn) / (tp + tn + fp + fn)
        elif self.__metric == "precision":
            return tp / (tp + fp)
        elif self.__metric == "recall":
            return tp / (tp + fn)
        elif self.__metric == "f1":
            return 2 * (tp / (tp + fp)) * (tp / (tp + fn)) / (tp / (tp + fn) + tp / (tp + fp))
        elif self.__metric == "roc_auc":
            dataset = pd.DataFrame({'y_proba': y_proba, 'pred_y': pred_y, }, columns=['y_proba', 'pred_y'])
            dataset.sort_values(by='y_proba', ascending=False, inplace=True)
            dataset["res1"] = dataset["pred_y"].cumsum().shift(1)

            mass = numpy.array([])

            for i in range(dataset.shape[0]):
                if dataset.iloc[i]["pred_y"] == 0:
                    res = dataset[(dataset["y_proba"] == dataset.iloc[i]["y_proba"]) & (dataset["pred_y"] == 1)]
                    mass = numpy.append(mass, res["pred_y"].cumsum())
                else:
                    mass = numpy.append(mass, 0)

            mass = np.nan_to_num(mass)
            mass /= 2
            res2 = pandas.Series(mass)

            dataset["res2"] = res2
            dataset["res1"].fillna(0, inplace=True)

            fir = dataset[dataset["pred_y"] == 0]["res1"].sum() + dataset[dataset["pred_y"] == 0]["res2"].sum()

            p = dataset["pred_y"].value_counts()[1]
            n = dataset["pred_y"].value_counts()[0]

            return round(fir / (p * n), 10)
        else:
            return None

data = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9), (7, 8, 9), (7, 8, 9), (7, 8, 92)],
                dtype=[("a", "i4"), ("b", "i4"), ("c", "i4")])
df3 = pd.DataFrame(data, columns=['c', 'a'])

model = MyLogReg(metric="roc_auc")
model.fit(x=df3, y=pandas.Series([1, 0, 1, 0, 0, 1]))





