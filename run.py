import argparse
import configparser
import os

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras.utils.np_utils import to_categorical


from model import model as sst_model

train_specInput_root_path = None
train_tempInput_root_path = None
train_label_root_path = None

test_specInput_root_path = None
test_tempInput_root_path = None
test_label_root_path = None

result_path = None
model_save_path = None


input_width = None
specInput_length = None
temInput_length = None

depth_spec = None
depth_tem = None
gr_spec = None
gr_tem = None
nb_dense_block = None
nb_class = None

nbEpoch = None
batch_size = None
lr = None


def read_config(config_path):
    conf = configparser.ConfigParser()
    conf.read(config_path)

    global train_specInput_root_path, train_tempInput_root_path, train_label_root_path, test_specInput_root_path, test_tempInput_root_path, test_label_root_path
    train_specInput_root_path = conf['path']['train_specInput_root_path']
    train_tempInput_root_path = conf['path']['train_tempInput_root_path']
    train_label_root_path = conf['path']['train_label_root_path']
    test_specInput_root_path = conf['path']['test_specInput_root_path']
    test_tempInput_root_path = conf['path']['test_tempInput_root_path']
    test_label_root_path = conf['path']['test_label_root_path']

    global result_path, model_save_path
    result_path = conf['path']['result_path']
    model_save_path = conf['path']['model_save_path']

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    global input_width, specInput_length, temInput_length
    input_width = int(conf['data']['input_width'])
    specInput_length = int(conf['data']['specInput_length'])
    temInput_length = int(conf['data']['temInput_length'])

    global depth_spec, depth_tem, gr_spec, gr_tem, nb_dense_block, nb_class
    depth_spec = int(conf['model']['depth_spec'])
    depth_tem = int(conf['model']['depth_tem'])
    gr_spec = int(conf['model']['gr_spec'])
    gr_tem = int(conf['model']['gr_tem'])
    nb_dense_block = int(conf['model']['nb_dense_block'])
    nb_class = int(conf['model']['nb_class'])

    global nbEpoch, batch_size, lr
    nbEpoch = int(conf['training']['nbEpoch'])
    batch_size = int(conf['training']['batch_size'])
    lr = float(conf['training']['lr'])


def run():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    K.set_image_data_format('channels_last')
    K.set_learning_phase(1)

    
    all_result_file = open(os.path.join(result_path, 'all_result.txt'), "w")
    all_result_file.close()
    
    for i in range(1, 16):
        all_result_file = open(os.path.join(result_path, 'all_result.txt'), "a")
        print('Subject:' + str(i))
        print('Subject ' + str(i) + ":", file=all_result_file)
        all_result_file.close()
        for j in range(1, 4):
            print("  Session:" + str(j))
            train_specInput = np.load(os.path.join(
                train_specInput_root_path, f"subject_{i}/section_{j}_data.npy"))
            train_tempInput = np.load(os.path.join(
                train_tempInput_root_path, f"subject_{i}/section_{j}_data.npy"))
            train_label = np.load(os.path.join(
                train_label_root_path, f"subject_{i}/section_{j}_data.npy"))

            index = np.arange(train_specInput.shape[0])
            np.random.shuffle(index)

            # print(train_specInput.shape)
            # print(train_tempInput.shape)
            # print(train_label.shape)

            train_specInput = train_specInput[index]
            train_tempInput = train_tempInput[index]
            train_label = train_label[index]
            
            print(train_specInput.shape)
            print(train_tempInput.shape)
            # print(train_label.shape)

            train_label = [x+1 for x in train_label]
            train_label = to_categorical(train_label, num_classes=3)
            print(train_label.shape)
            # print(train_label)

            # Evaluate
            test_specInput = np.load(os.path.join(
                test_specInput_root_path, f"subject_{i}/section_{j}_data.npy"))
            test_tempInput = np.load(os.path.join(
                test_tempInput_root_path, f"subject_{i}/section_{j}_data.npy"))

            test_label = np.load(os.path.join(
                test_label_root_path, f"subject_{i}/section_{j}_data.npy"))

            test_label = [x + 1 for x in test_label]
            test_label = to_categorical(test_label, num_classes=3)

            model = sst_model.sst_emotionnet(input_width=input_width, specInput_length=specInput_length, temInput_length=temInput_length,
                                            depth_spec=depth_spec, depth_tem=depth_tem, gr_spec=gr_spec, gr_tem=gr_tem, nb_dense_block=nb_dense_block, nb_class=nb_class)
            # model = multi_gpu_model(model, gpus=2)
            adam = keras.optimizers.Adam(
                lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
            model.compile(optimizer=adam, loss='categorical_crossentropy',
                          metrics=['accuracy'])

            early_stopping = EarlyStopping(
                monitor='val_loss', patience=20, verbose=1)

            save_model = ModelCheckpoint(
                filepath=os.path.join(model_save_path, f'Sub_{i}_Session_{j}.h5'),
                monitor='val_accuracy',
                save_best_only=True)
            history = model.fit([train_specInput, train_tempInput], train_label, epochs=nbEpoch, batch_size=batch_size,
                                validation_data=([test_specInput, test_tempInput], test_label), callbacks=[early_stopping, save_model], verbose=1)

            model = load_model(os.path.join(model_save_path, f'Sub_{i}_Session_{j}.h5'))
            loss, accuracy = model.evaluate(
                [test_specInput, test_tempInput], test_label)
            print('\ntest loss', loss)
            print('accuracy', accuracy)

            # Result Processing

            f = open(os.path.join(
                result_path, f'Sub_{i}_Session_{j}.txt'), "w")
            print(history.history, file=f)
            f.close()
            maxAcc = max(history.history['val_accuracy'])
            print("maxAcc = " + str(maxAcc))
            all_result_file = open(os.path.join(result_path, 'all_result.txt'), "a")
            print('  Session ' + str(j) + ":" +
                  str(accuracy), file=all_result_file)
            all_result_file.close()
            keras.backend.clear_session()
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument of running SST-EmotionNet.')
    parser.add_argument(
        '-c', type=str, help='Config file path.', required=True)
    args = parser.parse_args()
    read_config(args.c)
    run()
