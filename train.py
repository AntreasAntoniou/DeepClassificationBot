from keras.models import Sequential
import data
import numpy as np
import model as m


def run(epochs=150, split=0.1):
    '''Does the routine required to get the data, put them in needed format and start training the model
       saves weights whenever the model produces a better test result and keeps track of the best loss'''

    print("Extracting data...")
    X, y = data.extract_data()

    print("Getting data into shape...")
    data.preprocess_data(X, y, save=True)
    X, y = data.load_data()
    X_train, y_train, X_test, y_test = data.split_data(X, y, split_ratio=split)

    print("Building and Compiling model...")
    model = m.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    print("Training...")

    best_performance = np.inf
    for i in range(epochs):
        metadata = model.fit(X=X_train, y=y_train, batch_size=128, nb_epoch=1, verbose=1,
                             validation_data=[X_test, y_test])

        current_loss = metadata.history['loss'][-1]
        print("Loss: "+str(current_loss))

        if current_loss<best_performance:
            model.save_weights("pre_trained_weights/model_weights.h5df", overwrite=True)
            best_performance=current_loss
            print("Saving weights..")

if __name__ == '__main__':
    run()
