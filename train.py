from keras.models import Sequential
import data
import numpy as np
import model as m

def get_top_n_error(preds, y, n):
    index_of_true = np.argmax(y, axis=1)
    index_of_preds = np.argsort(preds, axis=1)
    total = len(y)
    correct = 0

    for i in range(len(index_of_true)):
        for j in range(1, n+1):
            if index_of_true[i] == index_of_preds[-j]:
                correct = correct+1
                break

    accuracy = correct/total

    return accuracy

def run(epochs=500, split=0.1, extract=False, cont=True):
    '''Does the routine required to get the data, put them in needed format and start training the model
       saves weights whenever the model produces a better test result and keeps track of the best loss'''
    if extract:
        print("Extracting data..")
        X, y = data.extract_data()

        print("Getting data into shape..")
        data.preprocess_data(X, y, save=True)

    print("Loading data..")
    X, y = data.load_data()
    X_train, y_train, X_test, y_test = data.split_data(X, y, split_ratio=split)

    print("Building and Compiling model..")
    print(y.shape)
    model = m.get_model(y_train.shape[1])

    if cont:
        model.load_weights("pre_trained_weights/model_weights.hdf5")
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    print("Training..")

    best_performance = np.inf
    for i in range(epochs):
        X_train_augmented = data.augment_data(X_train.copy())
        metadata = model.fit(X=X_train_augmented, y=y_train, batch_size=64, nb_epoch=1, verbose=1,
                             validation_data=[X_test, y_test], show_accuracy=True)
        n = 3
        current_loss = metadata.history['loss'][-1]
        current_val_loss = metadata.history['val_loss'][-1]
        preds = model.predict_proba(X_test, batch_size=64)
        print("Loss: "+str(current_loss))
        print("Val_loss: "+str(current_val_loss))

        top_3_error = get_top_n_error(preds, y_test, n)
        print("Top 3 error: "+str(top_3_error))
        if current_val_loss<best_performance:
            model.save_weights("pre_trained_weights/model_weights.hdf5", overwrite=True)
            best_performance=current_val_loss
            print("Saving weights..")

if __name__ == '__main__':
    run()

