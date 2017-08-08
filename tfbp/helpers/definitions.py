import os

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

data_dir = basedir + '/data'

global_step_path = data_dir + '/global_step.json'

dataset_path = data_dir + '/dataset.hdf5'

model_name = 'model.ckpt'
model_dir = data_dir + '/model'
model_path = '{}/{}'.format(model_dir, model_name)