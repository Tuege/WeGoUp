import math
import sys
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from stockstats import StockDataFrame
from stockstats import wrap, unwrap
import multiprocessing as mp
import time


def data_preprocessing(stock_data):
    stock_data.columns = stock_data.columns.str.lower()
    """
    stock_data = wrap(stock_data)
    # stock_data['macd']
    stock_data.init_all()
    stock_data = unwrap(stock_data)
    stock_data.dropna(inplace=True)
    """
    """
                     Open       High        Low      Close  Adj Close     Volume
    Date                                                                        
    2016-01-04  25.652500  26.342501  25.500000  26.337500  24.111500  270597600
    2016-01-05  26.437500  26.462500  25.602501  25.677500  23.507280  223164000
    2016-01-06  25.139999  25.592501  24.967501  25.174999  23.047245  273829600
    2016-01-07  24.670000  25.032499  24.107500  24.112499  22.074558  324377600
    2016-01-08  24.637501  24.777500  24.190001  24.240000  22.191275  283192000
    <class 'pandas.core.frame.DataFrame'>
    
    # print(type(stock_data.head()))
    """

    # plt.figure(figsize=(15, 8))
    # plt.title('Stock Prices History')
    # plt.plot(stock_data['close'])
    # plt.xlabel('Date')
    # plt.ylabel('Prices ($)')

    close_prices = stock_data['close']
    values = close_prices.values
    all_values = stock_data.values

    """
    all_values = np.array([[10, 20, 20, 10, 20, 20],
    [[10, 20, 20, 10, 20, 20],
    [20, 10, 30, 20, 10, 30],
    [30, 30, 10, 30, 30, 10],
    [40, 40, 510, 40, 40, 5010],
    [50, 210, 40, 50, 2010, 40],
    [110, 50, 50, 1010, 50, 50]]
    
    [[0.0, 0.05, 0.02, 0.0, 0.005, 0.002],
    [0.1, 0.0, 0.04, 0.01, 0.0, 0.004],
    [0.2, 0.1, 0.0, 0.02, 0.01, 0.0],
    [0.3, 0.15, 1.0, 0.03, 0.015, 1.0],
    [0.4, 1.0, 0.06, 0.04, 1.0, 0.006],
    [1.0, 0.2, 0.08, 1.0, 0.02, 0.008]]
    print(all_values)
    """
    # 1. Index: Time steps, 2. Index: Features
    """
    [ 26.33749962  25.67749977  25.17499924 ... 141.91000366 142.83000183
     141.5       ]
    <class 'numpy.ndarray'>
    
    # print(type(values))
    # print(values)
    """

    # calculate number of training samples needed (80% of entire dataset)
    training_data_len = math.ceil(all_values.shape[0] * 0.8)

    # Normalising the Data  between 0 and 1
    scaled_data = all_values.astype(np.float64)
    scaler_list = []

    for feature in range(scaled_data.shape[1]):
        scaler_list.append(MinMaxScaler(feature_range=(0, 1)))
        scaled_data[:, feature] = np.ravel(scaler_list[feature].fit_transform(all_values[:, feature].reshape(-1, 1)))

    """
    # Inverse Scaler Transform
    for feature in range(scaled_data.shape[1]):
        scaled_data[:,feature] = np.ravel(scaler_list[feature].inverse_transform(scaled_data[:,feature].reshape(-1,1)))
    """

    # split into training set and test set
    # scaled_data = scaled_data[4:]
    train_data = scaled_data[0:training_data_len, :]
    np.savetxt('data_printout.csv', train_data, delimiter=',')
    x_train = []
    y_train = []

    # shift the input and back to get the output array
    # create a 60 day window of data100 that is used to
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i])
        y_train.append(train_data[i, 0])

    # convert into a numpy
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = scaled_data[training_data_len:, 0]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
    return x_train, y_train, x_test, y_test, scaler_list, stock_data, training_data_len


def train(disp_queue: mp.Queue):
    stock_data = yf.download('AAPL', start='2016-01-01', end='2021-10-01')
    x_train, y_train, x_test, y_test, scaler_list, stock_data , training_data_len = data_preprocessing(stock_data)

    def scheduler(epoch, lr):
        if epoch < 1:
            return lr
        else:
            return lr * 1
            # print(lr * tf.math.exp(-0.1))
            # return lr * tf.math.exp(-0.1)

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    rmse_list = []
    error_list = []
    predictions_list = []
    x_train_original, y_train_original = x_train, y_train

    with tf.device('/GPU:0'):
        model = keras.Sequential()
        model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(layers.LSTM(100, return_sequences=False))
        model.add(layers.Dense(25))
        model.add(layers.Dense(1))
        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00003), loss='mean_squared_error')

        for training_shift in range(0, 1000, 1):
            x_train, y_train = x_train_original[training_shift:100 + training_shift], y_train_original[training_shift:100 + training_shift]
            model.fit(x_train, y_train, batch_size=1, epochs=5, callbacks=[callback])

            predictions = model.predict(x_test, verbose=False)

            predictions = np.ravel(scaler_list[0].inverse_transform(predictions.reshape(-1, 1)))
            target = np.ravel(scaler_list[0].inverse_transform(y_test.reshape(-1, 1)))
            rmse = np.sqrt(np.mean(predictions - target) ** 2)
            error = np.sum(np.absolute(predictions - target))
            predictions_list.append(predictions)
            rmse_list.append(rmse)
            error_list.append(error)
            if (error == min(error_list)) and (rmse < 5):
                print(training_shift)
                model.save('optimal_model2')
            disp_queue.put([rmse_list, target, predictions, error_list])
            print(rmse_list)

    data = stock_data.filter(['close'])
    train = data[:training_data_len]
    validation = data[training_data_len:]  # +4:]
    validation['predictions'] = predictions
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    line_list = []
    line_list.append(plt.plot(train, label='Train')[0])
    line_list.append(plt.plot(validation[['close']], label='Test')[0])
    plot_type = 'Important'
    match plot_type:
        case 'All':
            for plot in range(len(predictions_list)):
                validation.loc[str(plot)] = predictions_list[plot]
                line_list.append(plt.plot(validation[[str(plot)]], label=str(plot))[0])
        case 'Best':
            best_index = rmse_list.index(min(rmse_list))
            validation[str(best_index)] = predictions_list[best_index]
            line_list.append(plt.plot(validation[[str(best_index)]], label='Best Prediction')[0])
        case 'Final':
            line_list.append(plt.plot(validation[['predictions']], label='Final Prediction')[0])
        case 'Important':
            best_index = rmse_list.index(min(rmse_list))
            worst_index = rmse_list.index(max(rmse_list))
            best_error_index = error_list.index(min(error_list))
            validation[str(best_index)] = predictions_list[best_index]
            validation[str(worst_index)] = predictions_list[worst_index]
            validation[str(best_error_index)] = predictions_list[best_error_index]
            line_list.append(plt.plot(validation[[str(best_error_index)]], label='Best Prediction: Total Error')[0])
            line_list.append(plt.plot(validation[[str(best_index)]], label='Best Prediction: RMSE')[0])
            line_list.append(plt.plot(validation[[str(worst_index)]], label='Worst Prediction')[0])
            line_list.append(plt.plot(validation[['predictions']], label='Final Prediction')[0])

    plt.legend(handles=tuple(line_list), loc='lower right')
    plt.show()
    mp.current_process().terminate()


def run():
    stock_data = yf.download('AAPL', start='2016-01-01', end='2021-10-01')
    x_train, y_train, x_test, y_test, scaler_list, stock_data, training_data_len = data_preprocessing(stock_data)
    #y_train = y_train[training_data_len:]
    #x_train = x_train[training_data_len:]

    reconstructed_model = keras.models.load_model("optimal_model")

    with tf.device('/GPU:0'):
        t1 = time.time_ns()
        output = reconstructed_model.predict(x_train)
        output = np.ravel(scaler_list[0].inverse_transform(output.reshape(-1, 1)))
        t2 = time.time_ns()
    print((t2-t1)/1000000)

    plt.figure()
    plt.plot(output)
    target = np.ravel(scaler_list[0].inverse_transform(y_train.reshape(-1, 1)))
    plt.plot(target)
    rmse = np.sqrt(np.mean(output - target) ** 2)
    print(3*rmse)
    plt.fill_between(range(len(output)), output - 3*rmse, output + 3*rmse, alpha=0.2)
    diff = np.absolute(target - output)
    count = 0
    for n in diff:
        if n > 3*rmse:
            count += 1
    reliability = ((len(output)-count)*100)/len(output)
    print(reliability)
    plt.show()
