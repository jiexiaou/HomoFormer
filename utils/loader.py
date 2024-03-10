import os

from dataset import DataLoaderTrain, DataLoaderVal, DataLoaderTrain_h, DataLoaderVal_h, DataLoaderTrain_A_h, DataLoaderVal_A_h, DataLoaderSBUVal
def get_training_data(rgb_dir, img_options, plus=False):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None, plus=plus)

def get_validation_data(rgb_dir, plus=False):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, None, plus=plus)

def get_SBUvalidation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderSBUVal(rgb_dir, None)

def get_training_data_h(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain_h(rgb_dir, img_options, None)

def get_validation_data_h(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal_h(rgb_dir, None)

def get_training_data_A_h(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain_A_h(rgb_dir, img_options, None)

def get_validation_data_A_h(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal_A_h(rgb_dir, None)