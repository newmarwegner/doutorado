import os
import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import tensorflow as tf
from tensorflow import keras


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"

def filter_images(path, months, years=None):
    files = glob.glob(path + '*_ndvi.tif')
    order = lambda x: x.split('/')[-1].split('_')[0]
    files.sort(key=order)

    images = []
    for file in files:
        year_file = int(file.split('/')[-1].split('_')[0][0:4])
        month_file = int(file.split('/')[-1].split('_')[0][4:6])
        year_month = int(str(year_file) + str(month_file))
        if not years:
            if month_file in months:
                images.append(file)
        else:
            if year_month in years:
                images.append(file)

    return sorted(images, key=order)


def create_np_images(list_images, output_file):
    database = []
    for i in list_images:
        with rasterio.open(i) as dst:
            src = dst.read()
            database.append(src[0])

    with open(f'./templates/{output_file}.npy', 'wb') as f:
        np.save(f, database)


def split_sequence_2(sequence, n_steps_in):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = 1 + end_ix
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[i + 1:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


def create_shifted_frames(data):
    x = data[:, 0: data.shape[1] - 1, :, :]
    y = data[:, 1: data.shape[1], :, :]
    return x, y


def train_val_dataset(path_dataset, window):
    dataset = np.load(f'{path_dataset}')
    dataset, b = split_sequence_2(dataset, window)
    dataset = np.expand_dims(dataset, axis=-1)

    indexes = np.arange(dataset.shape[0])
    np.random.shuffle(indexes)
    train_index = indexes[: int(0.7 * dataset.shape[0])]
    val_index = indexes[int(0.7 * dataset.shape[0]):]

    return dataset[train_index], dataset[val_index]


def dataset_shapes(train_dataset, val_dataset):
    x_train, y_train = create_shifted_frames(train_dataset)
    x_val, y_val = create_shifted_frames(val_dataset)
    print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
    print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))

    return x_train, y_train, x_val, y_val


def create_model(x_train):
    inp = tf.keras.layers.Input(shape=(None, *x_train.shape[2:]))
    x = tf.keras.layers.ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = tf.keras.layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
    )(x)

    model = keras.models.Model(inp, x)

    model.compile(
        loss='mse', metrics=['mean_absolute_error'], optimizer=keras.optimizers.RMSprop(),
    )

    return model


def train_save(train_dataset, val_dataset, nm_model):
    x_train, y_train, x_val, y_val = dataset_shapes(train_dataset, val_dataset)
    model = create_model(x_train)
    # callbacks if whanna use
    # early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=10)
    epochs = 200
    batch_size = 5

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        # callbacks=[early_stopping, reduce_lr],
    )

    model.save(f'./templates/{nm_model}')

    with open(f'./templates/{nm_model}_history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    ##### evaluate model
    test_loss, test_acc = model.evaluate(x_val, y_val)
    print('Test acurácia: ', test_acc)

def plots(file_history):
    history_dict = pickle.load(open(f'./templates/{file_history}', "rb"))
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epochs = range(1, len(history_dict['loss']) + 1)

    plt.plot(epochs, loss_values, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    acc_values = history_dict['mean_absolute_error']
    val_acc_values = history_dict['val_mean_absolute_error']

    plt.plot(epochs, acc_values, '.', markersize=3, label='Training Acc')
    plt.plot(epochs, val_acc_values, 'b', linewidth=1, label='Validation Acc')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    ## Run to prepare and filter images considering month and year of production
    # ## variables
    # images_folder = '/home/newmar/Downloads/projetos_python/doutorado/indexes/'
    # month_global = [9, 10, 11, 12, 1, 2, 3]
    # monthyear_corn = [201812, 20191, 20192, 20193, 202011, 202012, 20211, 20212, 20213]
    # monthyear_soybean = [201911, 201912, 20201, 20202, 20203, 202111, 202112, 20221, 20222, 20223]
    #
    # ## filter datasets to use as Train dataset
    # list_images = filter_images(images_folder, month_global)
    # create_np_images(list_images, 'dados_global')

    ## Run to Create a Neural Network Model
    train_dataset, val_dataset = train_val_dataset('./templates/dados_global.npy', 10)
    train_save(train_dataset, val_dataset, 'global_10')
    #plots()


