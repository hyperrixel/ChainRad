"""
ChainRad
========

File: model training
"""


# Standard library imports
from os import listdir, mkdir
from os.path import isdir, isfile, join
import pickle
from tqdm import tqdm

# 3rd party imports
from PIL import Image
import torch

# Project level imports
from core import IMG_DIR, LOG_DIR, MODEL_DIR, OUT_DIR, SoloClassifier
from core import check_and_get_basics, get_accuracy, get_data_in_batches
from core import get_headless_models
from core import get_training_transformer


# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 5e-6
MAX_EPOCHS = 200

# Detecting device availability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    """
    Provides main functionality
    ===========================

    Raises
    ------
    RuntimeError
        When the folder of raw dataset images doesn't exist.
    """

    print('Device "{}" will be used for deep neural network operations.'
          .format(DEVICE))
    if not isdir(IMG_DIR):
        raise RuntimeError('Image folder missing, please download the dataset' +
                           ' or copy/move it to the IMG_DIR folder.')
    original_images = [f for f in listdir(IMG_DIR) if isfile(join(IMG_DIR, f))]
    if not isdir(OUT_DIR):
        mkdir(OUT_DIR)
    headless_outputs = [f for f in listdir(IMG_DIR) if isfile(join(OUT_DIR, f))]
    new_files = list(set(original_images).intersection(set(headless_outputs)))
    if len(new_files) > 0:
        print('{} files doesn\'t have headless output. Let\'s create them.'
              .format(len(new_files)))
        save_headless_outputs(new_files)
    train_binary_classifiers()


def save_headless_outputs(imagelist : list):
    """
    Save headless output of raw images
    ==================================

    Parameters
    ----------
    imagelist : list
        List of raw images to save as the concatenation of the outputs of the
        headless models.
    """

    # pylint: disable=no-member
    #         toch has a member function cat()
    #         Link: https://pytorch.org/docs/stable/generated/torch.cat.html

    headless = get_headless_models()
    for value in headless.values():
        value.to(DEVICE)
    transform = get_training_transformer()
    for filename in tqdm(imagelist, unit='image'):
        name_root = filename.split('.')[0]
        img = Image.open(join(IMG_DIR, filename)).convert('RGB')
        img = transform(img).unsqueeze(0).to(DEVICE)
        flats = []
        for value in headless.values():
            flats.append(value(img).squeeze(0).detach().cpu())
        final = torch.cat(flats)
        with open(join(OUT_DIR, '{}.out'.format(name_root)), 'wb') as outstream:
            pickle.dump(final, outstream)


def train_binary_classifiers():
    """
    Train binary classifiers
    ========================

    See also
    --------
        Error codes : check_prerequisites_and_get_basics()
    """

    # pylint: disable=too-many-statements
    #         Breaking this function to functions doesn't have too much sense.

    # pylint: disable=too-many-branches
    #         Breaking this function to functions doesn't have too much sense.

    # pylint: disable=too-many-locals
    #         Same variables are separated due to readability of the code, some
    #         other are needed being separated.

    # pylint: disable=no-member
    #         toch has a member functions round(), stack(), sigmoid()
    #         Link: https://pytorch.org/docs/stable/generated/torch.round.html
    #         Link: https://pytorch.org/docs/stable/generated/torch.stack.html
    #         Link: https://pytorch.org/docs/stable/generated/torch.sigmoid.html

    # pylint: disable=not-callable
    #         toch.tensor() is callable
    #         Link: https://pytorch.org/docs/stable/generated/torch.tensor.html

    diseases_basics = check_and_get_basics()
    for disease, meta_file_id in diseases_basics.items():
        print('\rDisease: {} --- initializing model...        '.format(disease),
              end='')
        disease_classifier = SoloClassifier()
        disease_classifier.to(DEVICE)
        optimizer = torch.optim.Adam(disease_classifier.parameters(),
                                     lr=LEARNING_RATE)
        criterion = torch.nn.BCEWithLogitsLoss()
        print('\rDisease: {} --- creating datasets...         '.format(disease),
              end='')
        train_dataset = get_data_in_batches(meta_file_id, batch_size=BATCH_SIZE)
        test_dataset = get_data_in_batches(meta_file_id, dataset_type='test',
                                           batch_size=1, shuffle_count=1)
        print('\rDisease: #{} --- training...                 '.format(disease),
              end='', flush=True)
        train_len = len(train_dataset)
        test_len = len(test_dataset)
        min_test_loss = 100.0
        test_no_decrease_count = 0
        with open(join(LOG_DIR, '{}.csv'.format(meta_file_id)), 'w',
                  encoding='utf8') as outstream:
            outstream.write('\t'.join(['epoch', 'train_loss', 'train_accuracy',
                                       'test_loss', 'test_accuracy']) + '\n')
        for epoch in range(MAX_EPOCHS):
            epoch_loss = 0.0
            epoch_preds, epoch_targets = [], []
            disease_classifier.train()
            torch.cuda.empty_cache()
            for batch_x, batch_y in tqdm(train_dataset, unit='batch',
                                         total=train_len):
                for _y in batch_y:
                    epoch_targets.append(_y)
                batch_x = torch.stack(batch_x).to(DEVICE)
                batch_y = torch.tensor(batch_y).float().to(DEVICE)
                optimizer.zero_grad()
                batch_y_hat = disease_classifier(batch_x)
                batch_y_hat = batch_y_hat.squeeze(1)
                loss = criterion(batch_y_hat, batch_y)
                loss.backward()
                optimizer.step()
                batch_loss = loss.item()
                epoch_loss += batch_loss
                batch_y_hat = torch.sigmoid(batch_y_hat)
                batch_y_hat = torch.round(batch_y_hat).tolist()
                for _y_hat in batch_y_hat:
                    epoch_preds.append(int(_y_hat))
            epoch_accuracy = get_accuracy(epoch_preds, epoch_targets)
            epoch_loss /= train_len
            print('{} TRAIN {:3d}/{:3d}: loss {:.9f} -- accuracy {:5.2f} %'
                  .format(disease, epoch + 1, MAX_EPOCHS, epoch_loss,
                          epoch_accuracy * 100))
            torch.save(disease_classifier.state_dict(),
                       join(MODEL_DIR, '{}_{:03d}.statedict'.format(meta_file_id,
                                                                    epoch + 1)))
            test_loss = 0.0
            test_preds, test_preds_float, test_targets = [], [], []
            torch.cuda.empty_cache()
            disease_classifier.eval()
            with torch.no_grad():
                for batch_x, batch_y in tqdm(test_dataset, unit='batch',
                                             total=test_len):
                    for _y in batch_y:
                        test_targets.append(_y)
                    batch_x = torch.stack(batch_x).to(DEVICE)
                    batch_y = torch.tensor(batch_y).float().to(DEVICE)
                    batch_y_hat = disease_classifier(batch_x)
                    batch_y_hat = batch_y_hat.squeeze(1)
                    loss = criterion(batch_y_hat, batch_y)
                    batch_loss = loss.item()
                    test_loss += batch_loss
                    batch_y_hat = torch.sigmoid(batch_y_hat)
                    batch_preds_float = batch_y_hat.tolist()
                    for _y_hat in batch_preds_float:
                        test_preds_float.append(float(_y_hat))
                    batch_y_hat = torch.round(batch_y_hat).tolist()
                    for _y_hat in batch_y_hat:
                        test_preds.append(int(_y_hat))
            test_accuracy = get_accuracy(test_preds, test_targets)
            test_loss /= test_len
            print('{} TEST {:3d}/{:3d}: loss {:.9f} -- accuracy {:5.2f} %'
                  .format(disease, epoch + 1, MAX_EPOCHS, test_loss,
                          test_accuracy  * 100),
                  flush=True)
            with open(join(LOG_DIR, '{}.csv'.format(meta_file_id)), 'a',
                      encoding='utf8') as outstream:
                outstream.write('{}\t{}\t{}\t{}\t{}\n'.format(epoch + 1,
                                epoch_loss, epoch_accuracy, test_loss,
                                test_accuracy))
            with open(join(LOG_DIR, 'last_{}.csv'.format(meta_file_id)), 'w',
                      encoding='utf8') as outstream:
                outstream.write('prediction\ttarget\n')
                for _x, _y in zip(test_preds_float, test_targets):
                    outstream.write('{}\t{}\n'.format(_x, _y))
            disease_classifier.load_state_dict(torch.load(join(MODEL_DIR,
                    '{}_{:03d}.statedict'.format(meta_file_id, epoch + 1))))
            if test_loss <= min_test_loss:
                test_no_decrease_count = 0
                min_test_loss = test_loss
            else:
                test_no_decrease_count += 1
            if test_no_decrease_count > 10:
                break
    print('Training finished.')


if __name__ == '__main__':
    main()
