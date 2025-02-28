import yaml
from keras.models import load_model
from dataIO.data import IonoDataManager
import numpy as np
from dataIO.dataPostprocess import get_minH_maxF
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cfgs = yaml.load(open('example_config.yaml','r'), Loader=yaml.BaseLoader)
dataManager = IonoDataManager(cfgs)

model = load_model('your_model_path')
res_mat = np.zeros([len(dataManager.test_data_list), 3, 6])
for idx in range(len(dataManager.test_data_list)):
    test_data, human_res, artist_res = dataManager.get_test_batch(idx)
    Dias_res = model.predict(test_data)
    res_mat[idx, :, 0:2] = get_minH_maxF(human_res)
    res_mat[idx, :, 2:4] = get_minH_maxF(artist_res)
    res_mat[idx, :, 4:6] = get_minH_maxF(Dias_res)

    if idx % 100 == 0:
        print(res_mat[idx, :])
np.save('save_path_name.npy', res_mat)