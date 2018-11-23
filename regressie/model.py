import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from sklearn import linear_model, preprocessing, model_selection

scaler = preprocessing.StandardScaler()

boston: DataFrame = pd.read_csv('./dataset/boston.csv')
feature_names = list(boston.columns.values)

boston_n = scaler.fit_transform(boston.values)

X = scaler.fit_transform(boston_n[:, :-1])
y = boston_n[:, -1]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)

def plot_boston():
    index = 0
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-white')
    for r in range(1, 6):
        for c in range(1, 5):
            if index >= X.shape[1]:
                break
            plt.subplot(4, 4, index + 1)
            plt.title(feature_names[index])
            plt.scatter(X[:, index], y, s=10)
            index += 1

    plt.show()
    plt.close()


def plot_boxplot():
    index = 0
    plt.figure(figsize=(20, 10))
    plt.style.use('seaborn-white')
    for k, v in boston.items():
        print('plotting', feature_names[index])
        plt.subplot(2, 7, index + 1)
        plt.title(feature_names[index])
        sns.boxplot(y=k, data=boston)
        index += 1

    plt.show()
    plt.close()


def plot_histplot():
    index = 0
    plt.figure(figsize=(20, 10))
    plt.style.use('seaborn-white')
    for k, v in boston.items():
        print('plotting', feature_names[index])
        plt.subplot(2, 7, index + 1)
        plt.title(feature_names[index])
        sns.distplot(v)
        index += 1

    plt.show()
    plt.close()


def plot_heatmap():
    plt.figure(figsize=(20, 10))
    sns.heatmap(boston.corr().abs(), annot=True)
    plt.show()
    plt.close()


def plot_lineair():
    min_max_scaler = preprocessing.MinMaxScaler()
    column_select = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
    x = boston.loc[:, column_select]
    y = boston['MEDV']
    x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_select)
    index = 0
    plt.figure(figsize=(20, 10))
    for i, k in enumerate(column_select):
        plt.subplot(2, 4, index + 1)
        sns.regplot(y=y, x=x[k], color=sns.color_palette()[index])
        index += 1
    plt.show()
    plt.close()


def regression():
    kf = model_selection.KFold(n_splits=10)

    linReg = linear_model.LinearRegression()
    scores = model_selection.cross_val_score(linReg, X_train, y, cv=kf, scoring='neg_mean_squared_error')
    print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

    linRegRegularisation = linear_model.Ridge()
    scores = model_selection.cross_val_score(linRegRegularisation, X_train, y, cv=kf, scoring='neg_mean_squared_error')
    print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))


def regularisation():
    kf = model_selection.KFold(n_splits=5)

    a_score = []
    a_values = []

    for a in range(1000):
        a = a / 10
        ridge = linear_model.Ridge(alpha=a)
        scores = model_selection.cross_val_score(ridge,
                                                 X,
                                                 y,
                                                 cv=kf,
                                                 scoring='neg_mean_squared_error')
        a_score.append(scores.mean())
        a_values.append(a)

    plt.plot(a_values, a_score)
    plt.show()
    plt.close()


if __name__ == '__main__':
    regularisation()
