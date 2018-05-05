# -*- coding: utf-8 -*-

# 多层感知机的softmax模型
def softmaxOutput():
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.optimizers import SGD
    import keras

    # Generate dummy data
    import numpy as np
    x_train = np.random.random((1000, 20))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
    x_test = np.random.random((100, 20))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(64, activation='relu', input_dim=20))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # 动量batch下降,lr学习率,decay:每轮学习后学习率的衰减,nesterov使用牛顿动量
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=20,
              batch_size=128)
    scores = model.evaluate(x_test, y_test, batch_size=128)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    return model


# 保存模型
def saveModel(_model):
    from keras.models import load_model
    from keras.models import model_from_json
    # creates a HDF5 file 'my_model.h5'
    _model.save('my_model.h5')
    # returns a compiled model
    # identical to the previous one
    model = load_model('my_model.h5')

    #只是希望保存模型的结构，而不包含其权重或配置信息
    # save as JSON
    json_string = model.to_json()
    model = model_from_json(json_string)

    print "model save finished"

# 如何获取中间层的输出
def getMiddleLayerOutput(model):
    import numpy as np
    from keras import backend as K
    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[2].output])
    x_test = np.random.random((100, 20))
    layer_output = get_3rd_layer_output([x_test])[0]
    print layer_output

#当验证集的loss不再下降时,使用Early_stopping
def earlyStopping(model,X,y):
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss',patience=2)
    # 选取数据最末尾的10%作为验证集
    model.fit(X,y,validation_split=0.1,callbacks=[early_stopping])

#使用census.csv训练多层神经网络
def censusSoftMax():
    from GetData import getData
    X_train, X_val, X_test, y_train, y_val, y_test = getData()
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import SGD
    import keras

    # Generate dummy data
    y_train = keras.utils.to_categorical(y_train, num_classes=2)
    y_test = keras.utils.to_categorical(y_train, num_classes=2)

    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(64, activation='relu', input_dim=103))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    # 动量batch下降,lr学习率,decay:每轮学习后学习率的衰减,nesterov使用牛顿动量
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(X_train, y_train)
              # epochs=20,
              # batch_size=128)
    scores = model.evaluate(X_train, y_train, batch_size=128)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


if __name__=="__main__":
    model = censusSoftMax()
    # getMiddleLayerOutput(model)

