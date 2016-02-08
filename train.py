from keras.models import Sequential
import data
import numpy as np
import model as m


def run(epochs=150, split=0.1):

    print("Extracting data...")
    X, y = data.extract_data()

    print("Getting data into shape...")
    X, y = data.preprocess_data(X, y)
    X_train, y_train, X_test, y_test = data.split_data(X, y, split_ratio=split)

    print("Building and Compiling model...")
    model=m.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    print("Training...")
    best_performance=np.inf
    for i in range(epochs):
        metadata = model.fit(X=X_train, y=y_train, batch_size=128, nb_epoch=1, verbose=1, validation_data=[X_test, y_test])
        current_loss=metadata.history['loss'][-1]
        print("Loss: "+str(current_loss))
        if current_loss<best_performance:
            model.save_weights("model_weights.h5df", overwrite=True)
            best_performance=current_loss
            print("Saving weights..")
