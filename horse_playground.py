from dbm import DBM
from scipy import ndimage
import numpy as np
import cPickle
import sys
import train_rbm
import tensorflow as tf
from PIL import Image
import os

HORSE_PATH='weizmann_horse_db/figure_ground/'
DATASET = 'weizmann.pkl'
DATASET_SIZE=328
RESOLUTION = 32
def load_dataset(n):
    #dataset = cPickle.load(file(DATASET,'rb'))
    #print 'loaded dataset', DATASET, ":", dataset.shape
    #return dataset
    dataset = np.zeros((DATASET_SIZE, n, n, 1))
    for i in range(DATASET_SIZE):
        if (i+1)==209: continue # outlier
        img = ndimage.imread(HORSE_PATH+'horse%03d.jpg'%(i+1))
        idxs = np.nonzero(img)
        left, right = np.min(idxs[1]), np.max(idxs[1])
        top, bottom = np.min(idxs[0]), np.max(idxs[0])

        img = img[top:bottom+1, left:right+1]
        image = Image.fromarray(img)
        image = image.resize((n, n), Image.ANTIALIAS)
        img = np.array(list(image.getdata()))
        img = img/255.0
        img = img.reshape((n,n,1))
        dataset[i] = img
    dataset[208] = dataset[-1]
    return dataset[:-1]

def build_dcbm(layers, name):
    conv = 1
    fc = 1
    input_shape = [RESOLUTION, RESOLUTION, 1] if layers[0][0]=='conv' else [RESOLUTION*RESOLUTION]
    dbm = DBM(input_shape, name=name)
    for layer in layers:
        if layer[0] == 'conv':
            filter_shape = layer[1]
            stride = layer[2]
            dbm.add_conv_layer(filter_shape, stride, 'VALID', 'conv%d'%conv)
            conv += 1
        elif layer[0] == 'fc':
            dbm.add_fc_layer(layer[1], 'fc%d'%fc)
            fc += 1
    return dbm

if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print 'usage: python dbm.py pcd/cd cd-k [output_dir]'
        sys.exit()
    else:
        use_pcd = (sys.argv[1] == 'pcd')
        cd_k = int(sys.argv[2])
        output_dir = None if len(sys.argv) == 3 else sys.argv[3]

    train_xs = load_dataset(RESOLUTION)[:325]
    train_xs = train_xs.reshape((-1, RESOLUTION, RESOLUTION, 1))
    train_xs_fc = train_xs.reshape([-1, RESOLUTION*RESOLUTION])
    init = train_xs[:10]
    init_fc = init.reshape(10,RESOLUTION*RESOLUTION)
    # list of layer configurations to run
    # configs = [ (['conv',(5,5,1,64), (2,2)],
    #              ['conv',(5,5,64,64),(2,2)],
    #              ['fc', 28*28]),
    #             (['conv',(5,5,1,64), (2,2)],
    #              ['conv',(5,5,64,64),(2,2)],
    #              ['fc', 64*64]),
    #             (['conv',(5,5,1,64), (3,3)],
    #              ['conv',(5,5,64,64),(3,3)],
    #              ['fc',28*28]),
    #             (['conv',(5,5,1,64), (4,4)],
    #              ['conv',(5,5,64,64),(4,4)],
    #              ['fc',28*28]),
    #             (['conv',(5,5,1,64), (2,2)],
    #              ['conv',(5,5,64,128),(2,2)],
    #              ['fc',28*28]),
    #             (['conv',(5,5,1,128), (2,2)],
    #              ['conv',(5,5,128,128),(2,2)],
    #              ['fc',28*28]),
    #             (['conv',(3,3,1,64), (2,2)],
    #              ['conv',(3,3,64,64),(2,2)],
    #              ['fc',28*28]),
    #             (['conv',(5,5,1,28), (2,2)],
    #              ['conv',(5,5,28,28),(2,2)],
    #              ['conv',(3,3,28,64),(2,2)],
    #              ['conv',(3,3,64,64),(2,2)],
    #              ['fc',28*28]),
    #             (['conv',(4,4,1,64), (3,3)],
    #              ['conv',(4,4,64,128),(3,3)],
    #              ['conv',(2,2,128,64),(1,1)],
    #              ['fc',28*28]),
    # ]
    batch_size = 5
    lr = 1e-4 if use_pcd else 1e-4

    # configs = [(['fc',(1024)],['fc',(2048)],['fc',(2048)])]
    # configs = [ [['conv',(14,14,1,64),(2,2)],
                 # ['conv',(9,9,64,64),(2,2)],
                 # # ['conv',(5,5,64,128),(2,2)],
                 # ['fc',500]]]
    # fuck (2,2), fucker (4,4)
    # model_name = 'horse_fuck_64'
    # configs = [ [['conv',(16,16,1,64),(2,2)],
    #              ['conv',(7,7,64,128),(2,2)],
    #              ['fc',500]]]
    #
    # model_name = 'horse_deep'
    # configs = [ [['conv',(16,16,1,64),(3,3)],
    #              ['conv',(7,7,64,64),(2,2)],
    #              ['conv',(3,3,64,128),(1,1)],
    #              ['fc',1000]]]
    # model_name = 'horse_32_centered'
    # configs = [ [['conv',(12,12,1,64),(2,2)],
    #              ['conv',(5,5,64,128),(2,2)],
    #              ['fc',500]]]
    # model_name = 'horse_32_centered_cutoff'
    # configs = [ [['conv',(12,12,1,64),(2,2)],
    #              ['conv',(5,5,64,128),(2,2)],
    #              ['fc',200]]]
    # model_name = 'horse_32_centered_cutoff_10_3_b20'
    # configs = [ [['conv',(10,10,1,64),(2,2)],
    #              ['conv',(6,6,64,128),(2,2)],
    #              ['fc',200]]]
    # model_name = 'horse_32_centered_cutoff_8_3_b20'
    # configs = [ [['conv',(8,8,1,64),(2,2)],
    #              ['conv',(3,3,64,128),(2,2)],
    #              ['fc',200]]]
    # model_name = 'horse_32_centered_cutoff_5_3_b20'
    # configs = [ [['conv',(5,5,1,64),(2,2)],
    #              ['conv',(3,3,64,128),(2,2)],
    #              ['fc',200]]]
    # model_name = 'horse_32_centered_cutoff_6_3_3_b20'
    # configs = [ [['conv',(6,6,1,64),(2,2)],# 12*12
    #              ['conv',(3,3,64,64),(1,1)], # 9*9, same as one 12*12 layer
    #              ['conv',(3,3,64, 64),(1,1)],
    #              ['conv',(3,3,64,64),(1,1)], #
    #              ['fc',200]]]
    # model_name = 'horse_32_centered_cutoff_6_3_3_3_b20'
    # configs = [ [['conv',(6,6,1,64),(2,2)],# 12*12
    #              ['conv',(3,3,64,64),(1,1)], # 9*9, same as one 12*12 layer
    #              ['conv',(3,3,64,128),(1,1)],
    #              ['conv',(3,3,128,128),(1,1)], #
    #              ['fc',500]]]
    # model_name = 'horse_32_centered_cutoff_12_3_3_b20_reg'
    # configs = [ [['conv',(12,12,1,64),(2,2)],
    #              ['conv',(3,3,64,128),(1,1)],
    #              ['conv',(3,3,128,128),(1,1)],
    #              ['fc',500]]]
    # model_name = 'horse_32_centered_cutoff_12_5_b25_reg'
    # configs = [ [['conv',(12,12,1,32),(2,2)],
    #              ['conv',(5,5,32,64),(2,2)],
    #              ['fc',300]]]
    # model_name = 'horse_32_centered_cutoff_12_5_b5_reg'
    # configs = [ [['conv',(12,12,1,32),(2,2)],
    #              ['conv',(5,5,32,64),(2,2)],
    #              ['fc',300]]]
    model_name = 'horse_32_centered_cutoff_12,32_5_b5'
    configs = [ [['conv',(12,12,1,32),(2,2)],
                 ['conv',(5,5,32,64),(2,2)],
                 ['fc',500]]]
    for idx, config in enumerate(configs):
        print configs
        name = model_name + '_%d' % idx
        print name
        dbm = build_dcbm(config, name)
        dbm.print_network()
        # if not os.path.exists(output_dir+name):
        #     os.mkdir(output_dir+name)
        #exit()
        if config[0][0] == 'conv':
            train_rbm.train(dbm, train_xs, lr, 5000, batch_size, use_pcd, cd_k, RESOLUTION, output_dir+name,
                            verbose=False, init=init, cutoffs=[0.1*i for i in range(3,8)])
        else:
            train_rbm.train(dbm, train_xs_fc, lr, 5000, batch_size, use_pcd, cd_k, RESOLUTION, output_dir+name,
                            verbose=False, init=init_fc, cutoffs=[0.6, 0.7])
        tf.reset_default_graph()
