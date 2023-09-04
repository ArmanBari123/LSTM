def do_difference(data, time_diff=1):
    df = []
    for row in range(time_diff, len(data)):
        df.append(data[row] - data[row - time_diff])
    return np.array(df)

def invert_difference(differenced_data, y_values):
    undiff = []
    index_to_start = len(differenced_data) + 1

    for i in range(len(differenced_data)):
        undiff.append(differenced_data[i] + y_values[-(index_to_start - i)])
    return undiff
    

def do_scaling(data):

    scaler = MinMaxScaler()
    scaler = scaler.fit(data.reshape(-1,1))
    scaledf = scaler.transform(data.reshape(-1,1))
    scaledf = scaledf.reshape(scaledf.shape[0])
    return scaler, scaledf

def invert_scaling(scaler, data):
    return scaler.inverse_transform(data).reshape(data.shape[0])
        

# given we have values of closing price for past 5 days, we want to predict the price of next day
def generate_sequence(data, lookback, target_col=None):
    traindf = []
    targetdf = []
    for i in range(len(data)):
        till_index = i +lookback
        if till_index <= len(data)-1:
            traindf.append(data[i: i + lookback])
            if target_col == None:
                targetdf.append(data[i+ lookback])
            else:
                targetdf.append(data[i+ lookback][target_col])
    return np.array(traindf), np.array(targetdf)


def train_test_split(x, y, train_percent=0.8):
    test_start_index = round(train_percent*len(x))
    trainx, trainy, testx, testy = x[0:test_start_index], y[0:test_start_index], x[test_start_index:], y[test_start_index:]
    return trainx, trainy, testx, testy

def lstm_model(trainx, trainy, neurons, lookback, features, epochs,batch_size=1, activation_func = "tanh"):
    X, y = trainx, trainy
    model = Sequential()
    # model.add(LSTM(4, activation='relu', input_shape=(n_steps, n_features)))
    model.add(LSTM(neurons, batch_input_shape=(batch_size,lookback, features), stateful=True))
    model.add(Dense(1)) 
    model.compile(optimizer='adam', loss='mean_squared_error')
    # X = X.reshape((X.shape[0], X.shape[1], n_features))
    X = X.reshape((X.shape[0], X.shape[1], features)) 
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, shuffle=False)
    return history, model

def plot_model_results(model):
    loss = model.history["loss"]
    plt.plot(loss)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.show()
    # return loss

def bidirectional_model(trainx, trainy, input_neurons, hidden_neurons, lookback, features, epochs,batch_size=128, dropout=0.5):
    X, y = trainx, trainy
    model = Sequential()
    model.add(Bidirectional(LSTM(input_neurons,return_sequences=True, dropout=0.5, input_shape=(lookback, features))))
    model.add(Bidirectional(LSTM(hidden_neurons, dropout=0.5)))
    model.add(Dense(1))  
    model.compile(optimizer='adam', loss='mean_squared_error')
    X = X.reshape((X.shape[0], X.shape[1], features))
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, shuffle=False)
    return history, model

def GRU_model(trainx, trainy, input_neurons, hidden_neurons, lookback, features, epochs,batch_size=128, dropout=0.5):
    X, y = trainx, trainy
    model = Sequential()
    model.add(GRU (input_neurons, dropout=0.5, return_sequences = True, input_shape=(lookback, features)))
    model.add(GRU(hidden_neurons, dropout=0.5))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    X = X.reshape((X.shape[0], X.shape[1], features))
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, shuffle=False)
    return history, model

def predict_data(model, testx, features=1):
    predictx = testx.reshape(testx.shape[0], testx.shape[1], features)
    y_pred = model.predict(predictx,batch_size=1)
    return y_pred

def plot_predicted(testy, y_pred, original_y, model_name, scaler):
    
    y_pred_unscaled = scaler.inverse_transform(y_pred)
    testy_unscaled = scaler.inverse_transform(testy.reshape(testy.shape[0],1))
    testy_undiff = invert_difference(testy_unscaled,original_y)
    pred_y_undiff = invert_difference(y_pred_unscaled, original_y)
    plt.plot(testy_undiff, label="Actual")
    plt.plot(pred_y_undiff, label="Predicted")
    plt.legend(loc="best")
    plt.title("Model Name = " + str(model_name))

def evaluate_results(testy, y_pred, original_y, model_name, scaler):
    errors = y_pred - testy
    rmse = np.sqrt(np.square(errors).mean())
    mae = np.mean(np.abs(errors))
    print('Root Mean Square Error = ' + str(rmse))
    print("mean Absolute Error = " + str(mae))
    plot_predicted(testy, y_pred, original_y, model_name, scaler)