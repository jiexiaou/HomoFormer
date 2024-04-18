import os

from dataset import DataLoaderTrain, DataLoaderVal, DataLoaderSBUVal
def get_training_data(rgb_dir, img_options, plus=False):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None, plus=plus)

def get_validation_data(rgb_dir, plus=False):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, None, plus=plus)

def get_SBUvalidation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderSBUVal(rgb_dir, None)
