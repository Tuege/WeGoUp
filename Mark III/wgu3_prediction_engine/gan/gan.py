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
from keras import layers
from stockstats import StockDataFrame
from stockstats import wrap, unwrap
import multiprocessing as mp
import time
import scipy.stats as stats
from customModels import customModels as models


def data_preprocessing(stock_data=None):
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
    train_data = scaled_data[0:training_data_len, :]
    # np.savetxt('data_printout.csv', train_data, delimiter=',')
    # retrieved_train_data = np.genfromtxt('data_printout.csv', delimiter=',', dtype=np.float64)
    x_train = []
    y_train = []

    # shift the input and back to get the output array
    # create a 60-day window of data that is used to
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i])
        y_train.append(train_data[i, 0])

    # convert into a numpy array
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


def train(queues):
    stock_data = yf.download('AAPL', start='2016-01-01', end='2021-10-01')
    x_train, y_train, x_test, y_test, scaler_list, stock_data, training_data_len = data_preprocessing(stock_data)
    print(np.shape(x_train))

    queues['scaler_queue'].put(scaler_list)

    rmse_list = []
    error_list = []
    predictions_list = []
    batch_time_list = []
    x_train_original, y_train_original = x_train, y_train
    batch_size = 100

    with tf.device('/GPU:0'):
        # Create the discriminator
        discriminator = models.DiscriminatorModel()

        # Create the generator
        model = models.GeneratorModel(shape=(x_train.shape[1], x_train.shape[2]))
        model.summary()

        # Import the custom Metrics
        progress_metrics = [
            # model.CustomMetrics.Loss(),
            model.CustomMetrics.Epoch(),
            model.CustomMetrics.Epochs(),
            model.CustomMetrics.Batch(),
            model.CustomMetrics.Batches(),
        ]

        # Compile Model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00003), loss='mean_squared_error', metrics=progress_metrics)

        # Import the custom Callbacks
        scheduler_callback = model.CustomCallbacks.Scheduler()
        progress_callbacks = model.CustomCallbacks.Progress(queues=queues)

        if True:
            start_time = time.time()
            # x_train, y_train = x_train_original[training_shift:100 + training_shift], y_train_original[training_shift:100 + training_shift]

            model.fit(x_train, y_train, batch_size=batch_size, epochs=25, callbacks=[progress_callbacks, scheduler_callback])

            # gui_process = mp.Process(target=model.TrainingGui, args=(queues,))
            # gui_process.start()

            predictions = model.predict(x_test, verbose=False)
            predictions = np.ravel(scaler_list[0].inverse_transform(predictions.reshape(-1, 1)))
            target = np.ravel(scaler_list[0].inverse_transform(y_test.reshape(-1, 1)))
            rmse = np.sqrt(np.mean(predictions - target) ** 2)
            error = np.sum(np.absolute(predictions - target))
            predictions_list.append(predictions)
            rmse_list.append(rmse)
            error_list.append(error)
            end_time = time.time()

            # Save model if it's good
            if (error == min(error_list)) and (rmse < 1):
                print("model saved")
                #model.save('optimal_model3')
            batch_time = end_time-start_time
            batch_time_list.append(batch_time)
            # TODO: tim_queue.put(batch_time_list)
            #   disp_queue.put([rmse_list, target, predictions, error_list, batch_time_list])
            print(rmse_list)

    data = stock_data.filter(['close'])
    train = data[:training_data_len]
    validation = data[training_data_len:]
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


def run():
    stock_data = yf.download('AAPL', start='2016-01-01', end='2021-10-01')
    x_train, y_train, x_test, y_test, scaler_list, stock_data, training_data_len = data_preprocessing(stock_data)

    reconstructed_model = keras.models.load_model("optimal_model")

    with tf.device('/GPU:0'):
        t1 = time.time_ns()
        output = reconstructed_model.predict(x_train)
        output = np.ravel(scaler_list[0].inverse_transform(output.reshape(-1, 1)))
        t2 = time.time_ns()
    print('Inference time: ' + str((t2-t1)/1000000) + 'ms')

    plt.figure()
    plt.plot(output)
    target = np.ravel(scaler_list[0].inverse_transform(y_train.reshape(-1, 1)))
    plt.plot(target)
    rmse = np.sqrt(np.mean(output - target) ** 2)
    print('3xRMSE: ' + str(3*rmse))

    diff = target - output
    diff_abs = np.absolute(diff)
    std = np.std(target - output)
    plt.fill_between(range(len(output)), output - 1.5*std, output + 1.5*std, alpha=0.2)
    print('3xSTD: ' + str(3*std))
    print('1.5xSTD: ' + str(1.5*std))
    count = 0
    for n in diff_abs:
        if n > 1.5*std:
            count += 1
    reliability = ((len(output)-count)*100)/len(output)
    print('Reliability of error corridor (1.5std): ' + str(reliability) + '%')
    print(len(output))
    plt.figure()
    counts, bins = np.histogram(diff, bins=100)
    plt.hist(bins[:-1], bins, weights=counts*(5/len(output)))
    mu = np.mean(diff)
    sigma = std
    x = np.linspace(mu - 10 * sigma, mu + 10 * sigma, 100)
    plt.plot(x, np.mean(diff)*stats.norm.pdf(x, mu, sigma))
    plt.show()
