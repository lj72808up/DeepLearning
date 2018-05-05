# -*- coding: utf-8 -*-

def fullConnection():
    from GetData import getData
    from keras.layers import Input,Dense
    from keras.models import Model
    import keras

    X_train, X_val, X_test, y_train, y_val, y_test = getData()
    y_train = keras.utils.to_categorical(y_train, num_classes=2)
    y_test = keras.utils.to_categorical(y_train, num_classes=2)

    inputs = Input(shape=(103,))
    x = Dense(64,activation='relu')(inputs)
    x = Dense(64,activation='relu')(x)
    predictions = Dense(2,activation="softmax")(x)

    # a layer instance is callable on a tensor, and returns a tensor
    model = Model(inputs=inputs,outputs=predictions)
    model.compile(optimizer="rmsprop",
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    model.fit(X_train,y_train)

    x1 = X_test.iloc[0,:]
    y1 = y_test[0,:]
    

if __name__=="__main__":
    fullConnection()