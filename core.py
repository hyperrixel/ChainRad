"""
ChainRad
========

File: core elements
"""


# Standard library imports
from csv import reader
from json import load as json_load
from os.path import isdir, isfile, join
from pickle import load as pickle_load
from random import shuffle

# 3rd party imports
import torch
from torchvision import models, transforms


IMG_DIR = './img'
LOG_DIR = './log'
META_DIR = './metadata'
MODEL_DIR = './models'
OUT_DIR = './out'


class SoloClassifier(torch.nn.Module):
    """
    Provide generic model architecture for binray classification
    ============================================================
    """

    # pylint: disable=abstract-method
    #         However _forward_unimplemented is abstract, according to
    #         PyTorch's it is not necessarily to override.

    # pylint: disable=too-many-instance-attributes
    #         The amount of attributes is needed because of the functionality.


    def __init__(self):
        """
        Initialize the object
        =====================
        """

        super().__init__()
        self.fc1 = torch.nn.Linear(30368, 2048)
        self.fc2 = torch.nn.Linear(2048, 256)
        self.fc3 = torch.nn.Linear(256, 32)
        self.fc4 = torch.nn.Linear(32, 1)
        self.activation = torch.nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=True)


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Perform forward operation on the model
        ======================================

        Parameters
        ----------
        x : torch.Tensor
            Values to use for predicition.

        Returns
        -------
        torch.Tensor
            Predicted values.
        """

        # pylint: disable=invalid-name
        #         The use of name x accords to PyTorch's documentation.

        x = self.activation(self.dropout(self.fc1(x)))
        x = self.activation(self.dropout(self.fc2(x)))
        x = self.activation(self.dropout(self.fc3(x)))
        return self.fc4(x)


class EmptyLayer(torch.nn.Module):
    """
    Empty layer class to substitute classifier layer(s)
    ===================================================
    """

    # pylint: disable=abstract-method
    #         However _forward_unimplemented is abstract, according to
    #         PyTorch's it is not necessarily to override.


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward x without any change
        ============================

        Parameters
        ----------
        x : torch.Tensor
            Input values.

        Returns
        -------
        torch.Tensor
            Output values (input values unchanged).
        """

        # pylint: disable=no-self-use
        #         This method doesn't need to use self but to be a member method
        #         instead of being a function is essential.

        # pylint: disable=invalid-name
        #         The use of name x accords to PyTorch's documentation.

        return x


def check_and_get_basics():
    """
    Check train prerequisites and get diseases basics
    =================================================

    Raises
    ------
    RuntimeError
        When the metadata directory doesn't exist.
    FileNotFoundError
        When diseases information file doesn't exist.
    TypeError
        When disease information file has invalid data.
    FileNotFoundError
        Wwhen a metafile for a disease doesn't exist.
    """

    if not isdir(META_DIR):
        raise RuntimeError('Training is not possible without metadata folder.')
    if not isfile(join(META_DIR, 'diseases.json')):
        raise FileNotFoundError('Training is not possible without diseases ' +
                                'information.')
    with open(join(META_DIR, 'diseases.json'), 'r', encoding='utf8') as instream:
        result = json_load(instream)
    if not isinstance(result, dict):
        raise TypeError('Training is not possible because disease information' +
                        'should be a Dictionary but it is something else.')
    for disease, meta_file_prefix in result.items():
        if not isfile(join(META_DIR, 'train_{}.csv'.format(meta_file_prefix))):
            raise FileNotFoundError('Training is not possible without disease' +
                                    'training metafile for "{}".'
                                    .format(disease))
        if not isfile(join(META_DIR, 'test_{}.csv'.format(meta_file_prefix))):
            raise FileNotFoundError('Training is not possible without disease' +
                                    'test metafile for "{}".'
                                    .format(disease))
        if not isfile(join(META_DIR, 'valid_{}.csv'
                                     .format(meta_file_prefix))):
            raise FileNotFoundError('Training is not possible without disease' +
                                    'validation metafile for "{}".'
                                    .format(disease))
    return result


def get_accuracy(pred_list : list, target_list : list) -> float:
    """
    Get prediction's accuracy
    =========================

    Parameters
    ----------
    pred_list : list
        List of predicitions.
    target_list : list
        List of targets.

    Returns
    -------
    float
        The rate of good predictions.
    """

    good_count = 0
    for pred, target in zip(pred_list, target_list):
        if pred == target:
            good_count += 1
    return good_count / len(pred_list)


def get_data_in_batches(meta_file_id : str, dataset_type : str = 'train',
                        batch_size : int = 1, drop_last : bool = False,
                        shuffle_count : int = 3) -> list:
    """
    Get data in batches
    ===================

    Parameters
    ----------
    meta_file_id : str
        Identifier of the dataset to wowrk with.
    dataset_type : str, optional ('train' if omitted)
        Type of the dataset to work with. Common values are 'train', 'test',
        'valid'.
    batch_size : int, optional (1 if omitted)
        Size of individual batches.
    drop_last : bool, optional (False if omitted)
        Whether or not to drop the last batch if its size is less then
        the given batch size.
    shuffle_count : int, optional (3 if omitted)
        Number of shuffles to make on the dataset before batching.

    Raises
    ------
    FileNotFoundError
        WHen the given meta file with the given dataset type doesn't exist.
    """

    # pylint: disable=too-many-locals
    #         Same variables are separated due to readability of the code, some
    #         other are needed being separated.

    # pylint: disable=unused-variable
    #         However i is not used, it is required in the for loop.

    filename = join(META_DIR, '{}_{}.csv'.format(dataset_type, meta_file_id))
    if not isfile(filename):
        raise FileNotFoundError('Cannot find "{}".'.format(filename))
    content = []
    with open(filename, 'r', encoding='utf8') as instream:
        for row in list(reader(instream, delimiter='\t'))[1:]:
            with open(join(OUT_DIR, row[0].split('.')[0] + '.out'),
                      'rb') as instream:
                _x = pickle_load(instream)
            _y = int(row[3])
            content.append((_x, _y))
    for i in range(shuffle_count):
        shuffle(content)
    result = []
    pos = 0
    len_content = len(content)
    while pos + batch_size < len_content:
        x_list, y_list = [], []
        for _x, _y in content[pos:pos + batch_size]:
            x_list.append(_x)
            y_list.append(_y)
        result.append((x_list, y_list))
        pos += batch_size
    if pos < len_content - 1 and not drop_last:
        x_list, y_list = [], []
        for _x, _y in content[pos:]:
            x_list.append(_x)
            y_list.append(_y)
        result.append((x_list, y_list))
    return result


def get_headless_models() -> dict:
    """
    Create dict of headless models
    ==============================

    Returns
    -------
    dict
        Dictionary of headless models.
    """

    result = {}
    result['VGG16bn'] = models.vgg16_bn(pretrained=True)
    result['VGG16bn'].classifier = EmptyLayer()
    result['ResNet152'] = models.resnet152(pretrained=True)
    result['ResNet152'].fc = EmptyLayer()
    result['DenseNet161'] = models.densenet161(pretrained=True)
    result['DenseNet161'].classifier = EmptyLayer()
    result['GoogleNet'] = models.googlenet(pretrained=True)
    result['GoogleNet'].fc = EmptyLayer()
    for key in result.keys():
        for param in result[key].parameters():
            param.requires_grad = False
        result[key].eval()
    return result


def get_simple_transformer() -> transforms.transforms.Compose:
    """
    Get composed simple transformer
    ===============================

    Returns
    -------
    Compose
        A composition of transforms.
    """

    result = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=
                                                      [0.485, 0.456, 0.406],
                                                      std=
                                                      [0.229, 0.224, 0.225])])
    return result


def get_training_transformer(rotation_degree : any = 13,
                             translate : tuple = (0.1, 0.1),
                             shear : float = 0.1, scale : tuple = (0.9, 1.1),
                             h_flip_probability : float = 0.5
                             ) -> transforms.transforms.Compose:
    """
    Get composed training transformer
    =================================

    Parameters
    ----------
    rotation_degree : float | tuple, optional (13 if omitted)
        Degree to randomly rotate images. If a number is given, rotation is
        between - and + number. If a tuple with two elements is given,
        rotation is in the range of the two elements. If more than two elements
        is igiven, level of rotation is selected from the elements.
    translate : tuple, optional ((0.1, 0.1) if omitted)
        Maximum absolute fraction for horizontal and vertical random
        translations.
    shear : float | tuple, optional (0.1 if omitted)
        Random shear range. If a float is given, random shear is applied on the
        x axis between - and + number. If a two element tuple is given, random
        shear is applied on the x axis between the elements. If a four element
        tuple is given, shear is applied on the x acis between the first two,
        on the y axis between the last two elements.
    scale : tuple, optional ((0.9, 1.1) if omitted)
        Interval of random scale.
    h_flip_probability : float = 0.5,
        Probability of a random horizontal flip.

    Returns
    -------
    Compose
        A composition of transforms.
    """

    # pylint: disable=too-many-arguments
    #         We consider a better practice having long list of named arguments
    #         then having **kwargs only.

    result = transforms.Compose([transforms.RandomAffine(rotation_degree,
                                                         translate=translate,
                                                         shear=shear,
                                                         scale=scale),
                                 transforms.RandomHorizontalFlip(
                                                        p=h_flip_probability),
                                 transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=
                                                      [0.485, 0.456, 0.406],
                                                      std=
                                                      [0.229, 0.224, 0.225])])
    return result
