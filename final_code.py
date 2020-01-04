import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from pandas_datareader import data as pdr
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


def run_predictor(ticker, beginning_date):
    sp = StockPredictor()
    sp.getData(ticker, beginning_date)
    sp.preprocess()
    sp.normalize_values()
    sp.create_training_data()
    sp.train_benchmark_model()
    sp.train_rfr()
    sp.train_knn()
    sp.train_dtr()
    sp.train_svr()
    sp.train_gbr()


class StockPredictor:

    def getData(self, ticker, beginningDate):

        self.ticker = ticker
        self.beginning_date = beginningDate
        self.end_date = time.strftime('%Y-%m-%d')

        self.stock_data = pdr.get_data_yahoo(self.ticker, start=self.beginning_date, end=self.end_date)
        self.stock_data = self.stock_data.drop('Close', 1)

        ko = pdr.get_data_yahoo('KO', start=self.beginning_date, end=self.end_date)
        aapl = pdr.get_data_yahoo('AAPL', start=self.beginning_date, end=self.end_date)
        tsla = pdr.get_data_yahoo('TSLA', start=self.beginning_date, end=self.end_date)

        rollko = ko[['Adj Close']].rolling(10)
        rollaapl = aapl[['Adj Close']].rolling(10)
        rolltsla = tsla[['Adj Close']].rolling(10)

        # plot = plt.figure()
        # ax = plot.add_subplot(1, 1, 1)
        # ax.plot(range(len(self.ko)), rollko.std(ddof=0), label='KO')
        # ax.plot(range(len(self.ko)), rollaapl.std(ddof=0), label='AAPL')
        # ax.plot(range(len(self.ko)), rolltsla.std(ddof=0), label='TSLA')
        # ax.set_xlabel('Date')
        # ax.set_ylabel('10-Day Volatility value')
        #
        # plt.legend()
        # plt.title("'KO', 'AAPL', and 'TSLA' volatility comparision")
        # plt.show()

    def preprocess(self):
        target_df = pd.DataFrame()
        target_df["Shift"] = self.stock_data['Adj Close']
        target_df['Shift'].dropna(inplace=True)
        self.label = target_df['Shift']
        self.stock_data_copy = self.stock_data.drop('Adj Close', 1)

    def normalize_values(self):
        sc = StandardScaler()
        self.scaler = sc.fit(self.stock_data_copy)
        self.scaled_data = pd.DataFrame(self.scaler.transform(self.stock_data_copy))

    def create_training_data(self):

        # Code is based off of the sklearn documentation for TimeSeriesSplit

        tss = TimeSeriesSplit(n_splits=5)
        #
        print("scaled data: ")
        print(self.scaled_data.values)
        #
        for train_index, test_index in tss.split(self.scaled_data.values):

            self.X_train, self.X_test = self.scaled_data[train_index[0]:train_index[-1]], self.scaled_data[test_index[0]:test_index[-1]]
            self.y_train, self.y_test = self.label[train_index[0]:train_index[-1]], self.label[test_index[0]:test_index[-1]]


    def train_benchmark_model(self):
        self.rfr = RandomForestRegressor().fit(self.X_train, self.y_train)
        predicted = self.rfr.predict(self.X_test)

        print("The R2 score of the benchmark RFR is: " + str(r2_score(predicted, self.y_test)))
        print("The actual price for benchmark RFR: ")
        print(self.y_test.tail(1))
        print("The predicted price for benchmark RFR: ")
        print(predicted[-1])

    def train_rfr(self):
        params = {'n_estimators': range(5, 15)}
        rfr = RandomForestRegressor()
        grid_search = GridSearchCV(rfr, params)
        grid_search.fit(self.X_train, self.y_train)
        predicted = grid_search.predict(self.X_test)

        print("The R2 score of RFR is: " + str(r2_score(predicted, self.y_test)))
        print("The actual price for RFR: ")
        print(self.y_test.tail(1))
        print("The predicted price for RFR: ")
        print(predicted[-1])

    def train_knn(self):
        params = {'n_neighbors': range(1, 11), 'weights': ['uniform', 'distance'],
                  'algorithm': ['ball_tree', 'kd_tree', 'brute'], 'leaf_size': range(10, 40)}
        knn = KNeighborsRegressor()
        grid_search = GridSearchCV(knn, params)
        grid_search.fit(self.X_train, self.y_train)
        predicted = grid_search.predict(self.X_test)

        print("The R2 score of KNN is: " + str(r2_score(predicted, self.y_test)))
        print("The actual price for KNN: ")
        print(self.y_test.tail(1))
        print("The predicted price for KNN: ")
        print(predicted[-1])

    def train_dtr(self):
        params = {'min_samples_split': [2, 3, 4, 5], 'min_samples_leaf': range(1, 5)}
        dtr = DecisionTreeRegressor()
        grid_search = GridSearchCV(dtr, params)
        grid_search.fit(self.X_train, self.y_train)
        predicted = grid_search.predict(self.X_test)

        print("The R2 score of DTR is: " + str(r2_score(self.y_test, predicted)))
        print("The actual price for DTR: ")
        print(self.y_test.tail(1))
        print("The predicted price for DTR: ")
        print(predicted[-1])

    def train_svr(self):
        params = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 'C': range(1, 11)}
        svr = SVR()
        grid_search = GridSearchCV(svr, params)
        grid_search.fit(self.X_train, self.y_train)
        predicted = grid_search.predict(self.X_test)

        print("The R2 score of SVR is: " + str(r2_score(self.y_test, predicted)))
        print("The actual price for SVR: ")
        print(self.y_test.tail(1))
        print("The predicted price for SVR: ")
        print(predicted[-1])

    def train_gbr(self):
        params = {'loss': ['ls', 'lad', 'huber', 'quantile'],
                  'max_depth': range(1, 11)}
        gbr = GradientBoostingRegressor()
        grid_search = GridSearchCV(gbr, params)
        grid_search.fit(self.X_train, self.y_train)
        predicted = grid_search.predict(self.X_test)

        print("The R2 score of GBR is: " + str(r2_score(self.y_test, predicted)))
        print("The actual price for GBR: ")
        print(self.y_test.tail(1))
        print("The predicted price for GBR: ")
        print(predicted[-1])
