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
DATASET_SIZE=288

def load_dataset(n):
    #dataset = cPickle.load(file(DATASET,'rb'))
    #print 'loaded dataset', DATASET, ":", dataset.shape
    #return dataset
    dataset = np.zeros((DATASET_SIZE, n, n, 1))
    for i in range(DATASET_SIZE):
        img = Image.open(HORSE_PATH+'horse%03d.jpg'%(i+1))
        img = img.resize((n, n), Image.ANTIALIAS)
        img = np.array(list(img.getdata()))
        img = img/255.0
        img = img.reshape((n,n,1))
        dataset[i] = img
    return dataset

def build_dcbm(layers, name):
    conv = 1
    fc = 1
    input_shape = [28, 28, 1] if layers[0][0]=='conv' else [28*28]
    dbm = DBM(input_shape, name=name)
    for layer in layers:
        if layer[0] == 'conv':
            filter_shape = layer[1]
            stride = layer[2]
            dbm.add_conv_layer(filter_shape, stride, 'SAME', 'conv%d'%conv)
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

    train_xs = load_dataset(28)[:280]
    train_xs = train_xs.reshape((-1, 28, 28, 1))
    train_xs_fc = train_xs.reshape([-1, 28*28])
    batch_size = 20
    lr = 0.001 if use_pcd else 0.1

    configs = [ (['conv',(16,16,1,500), (14,14)],
                 ['fc', 100])]
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
    # configs = [(['fc',(1024)],['fc',(2048)],['fc',(2048)])]
    for idx, config in enumerate(configs):
        print configs
        name = 'dbm_%d' % idx
        dbm = build_dcbm(config, name)
        dbm.print_network()
        # if not os.path.exists(output_dir+name):
        #     os.mkdir(output_dir+name)
        if config[0][0] == 'conv':
            train_rbm.train(dbm, train_xs, lr, 50, batch_size, use_pcd, cd_k, output_dir+name)
        else:
            train_rbm.train(dbm, train_xs_fc, lr, 50, batch_size, use_pcd, cd_k, output_dir+name)
        tf.reset_default_graph()
