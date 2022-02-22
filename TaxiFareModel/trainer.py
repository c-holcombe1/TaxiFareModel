from TaxiFareModel.data import get_data, clean_data

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

from TaxiFareModel.encoders import TimeFeaturesEncoder
from TaxiFareModel.encoders import DistanceTransformer

from TaxiFareModel.utils import compute_rmse

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y



    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
         ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
         ])
        preproc_pipe = ColumnTransformer([
         ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
         ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        self.pipe = Pipeline([
         ('preproc', preproc_pipe),
         ('linear_model', LinearRegression())
         ])
        return self.pipe

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipe.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipe.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse


if __name__ == "__main__":
    # get data
    df=get_data()
    print(df)
    # clean data
    df=clean_data(df)
    print(df)
    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    print(X,y)
    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    # train
    trainer=Trainer(X_train,y_train)
    print(trainer)
    trainer.run()

    # evaluate
    rmse=trainer.evaluate(X_val,y_val)

    print(rmse)
