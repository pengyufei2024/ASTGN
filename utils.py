import numpy as np
import pandas as pd

def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape

def seq2instance(data, M, N):
    num_step, dims = data.shape
    num_sample = num_step - M - N + 1
    x = np.zeros(shape = (num_sample, M, dims))
    y = np.zeros(shape = (num_sample, N, dims))
    for i in range(num_sample):
        x[i] = data[i : i + M]
        y[i] = data[i + M : i + M + N]
    return x, y

def loadData(args):
    df = pd.read_hdf(args.traffic_file)
    Traffic = df.values
    num_step = df.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = Traffic[: train_steps]
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps :]
    trainX, trainY = seq2instance(train, args.M, args.N)
    valX, valY = seq2instance(val, args.M, args.N)
    testX, testY = seq2instance(test, args.M, args.N)
    mean, std = np.mean(trainX), np.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std
    f = open(args.SE_file, mode = 'r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape = (N, dims), dtype = np.float32)
    for line in lines[1 :]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index-1] = temp[1 :]
    Time = df.index
    dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
                // (5*60)
    timeofday = np.reshape(timeofday, newshape = (-1, 1))    
    Time = np.concatenate((dayofweek, timeofday), axis = -1)
    train = Time[: train_steps]
    val = Time[train_steps : train_steps + val_steps]
    test = Time[-test_steps :]
    trainTE = seq2instance(train, args.M, args.N)
    trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
    valTE = seq2instance(val, args.M, args.N)
    valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
    testTE = seq2instance(test, args.M, args.N)
    testTE = np.concatenate(testTE, axis = 1).astype(np.int32)
    
    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
            SE, mean, std)
