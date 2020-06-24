from sklearn.preprocessing import MinMaxScaler
import joblib

class scaler:
    def __init__(self, df):
        self.__scalername = ".save"
        self.data = df

    #clean data on train proccess and save tha scaler
    def clean_train(self, column_to_scaler):
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.data[column_to_scaler] = scaler.fit_transform(self.data[column_to_scaler].values.reshape(-1, 1))
        joblib.dump(scaler, column_to_scaler + self.__scalername)

    # load data scaler used on train to predict proccess
    def clean_predict(self, column_to_scaler):
        scalerPredict = joblib.load(column_to_scaler + self.__scalername)
        self.data[column_to_scaler] = scalerPredict.transform(self.data[column_to_scaler].values.reshape(-1, 1))
