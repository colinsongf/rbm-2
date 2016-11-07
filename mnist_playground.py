from dbm import DBM
import numpy as np
import cPickle
import sys
import train_rbm
import tensorflow as tf

DATASET = 'mnist.pkl'

def load_dataset():
    (train_xs, _), _, _ = cPickle.load(file('mnist.pkl', 'rb'))
    init = cPickle.load(file('testdigits.pkl','rb'))
    print init.shape
    return train_xs, init

def build_dcbm(layers, name):
    conv = 1
    fc = 1
    input_shape = [28, 28, 1] if layers[0][0]=='conv' else [28*28]
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
    dbm.print_network()
    return dbm

if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print 'usage: python dbm.py pcd/cd cd-k [output_dir]'
        sys.exit()
    else:
        use_pcd = (sys.argv[1] == 'pcd')
        cd_k = int(sys.argv[2])
        output_dir = None if len(sys.argv) == 3 else sys.argv[3]

    train_xs, init = load_dataset()
    train_xs = train_xs.reshape((-1, 28, 28, 1))
    train_xs_fc = train_xs.reshape((-1, 28*28))
    init = init.reshape((-1, 28, 28, 1))
    init_fc = init.reshape(-1, 28*28)
    batch_size = 40
    lr = 1e-4 if use_pcd else 0.1
    '''
    models = [ \
                (['conv',(5,5,1,64), (2,2)], #0
                 ['conv',(5,5,64,64),(2,2)],
                 ['fc', 1024]),
                (['conv',(5,5,1,64), (3,3)], #1
                 ['conv',(5,5,64,64),(3,3)],
                 ['fc',1024]),
                (['conv',(5,5,1,64), (4,4)], #2
                 ['conv',(5,5,64,64),(4,4)],
                 ['fc',1024]),
                (['conv',(5,5,1,64), (5,5)], #3
                 ['conv',(5,5,64,64),(5,5)],
                 ['fc',1024]),
                (['conv',(5,5,1,64), (2,2)], #4
                 ['conv',(5,5,64,64),(2,2)],
                 ['fc', 1024]),
                (['conv',(5,5,1,64), (2,2)], #5
                 ['conv',(5,5,64,128),(2,2)],
                 ['fc',1024]),
                (['conv',(5,5,1,128), (2,2)], #6
                 ['conv',(5,5,128,128),(2,2)],
                 ['fc',1024]),
                (['conv',(3,3,1,64), (1,1)], #7
                 ['conv',(3,3,64,64),(1,1)],
                 ['fc',1024]),
                (['conv',(3,3,1,64), (2,2)], #8
                 ['conv',(3,3,64,64),(2,2)],
                 ['fc',1024]),
                (['conv',(3,3,1,64), (3,3)], #9
                 ['conv',(3,3,64,64),(3,3)],
                 ['fc',1024]),
                (['conv',(5,5,1,32), (2,2)], #10
                 ['conv',(5,5,32,32),(2,2)],
                 ['conv',(3,3,32,64),(2,2)],
                 ['conv',(3,3,64,128),(2,2)],
                 ['fc',1024]),
                (['conv',(3,3,1,32), (2,2)], #11
                 ['conv',(3,3,32,32),(2,2)],
                 ['conv',(5,5,32,64),(2,2)],
                 ['fc',1024]),
                (['conv',(5,5,1,64), (2,2)], #12
                 ['conv',(5,5,64,128),(2,2)],
                 ['conv',(5,5,128,128),(2,2)],
                 ['fc',1024]),
                (['conv',(4,4,1,64), (3,3)], #13
                 ['conv',(4,4,64,128),(3,3)],
                 ['conv',(4,4,128,128),(3,3)],
                 ['fc',1024])
    ]

    models = [
        (['conv',(12,12,1,64), (2,2)], #0
         ['conv',(5,5,64,128),(2,2)],
         ['fc', 500]),
        (['conv',(12,12,1,64), (2,2)], #0
         ['conv',(5,5,64,64),(2,2)],
         ['fc', 500]),
        (['conv',(12,12,1,64), (4,4)], #2
         ['conv',(3,3,64,128),(2,2)],
         ['fc', 500]),
        (['conv',(14,14,1,64), (2,2)], #3
         ['conv',(4,4,64,128),(2,2)], # 8*8
         ['fc', 500]),
        (['conv',(10,10,1,64), (2,2)], #4
         ['conv',(6,6,64,128),(2,2)], # 10*10
         ['fc', 500]),
        (['conv',(10,10,1,64), (2,2)], #5
         ['conv',(4,4,64,128),(2,2)], # 10*10
         ['fc', 500])
    ]

    name = 'dbm_128'
    dbm = DBM([28, 28, 1], name=name)
    dbm.add_conv_layer((12,12,1,128), (2,2), 'VALID', 'conv1')
    dbm.add_conv_layer((5,5,128,256), (2,2), 'VALID', 'conv2')
    dbm.add_fc_layer(1000,'fc1')
    dbm.print_network()
    train_rbm.train(dbm, train_xs, lr, 60, batch_size, use_pcd, cd_k, output_dir+name, init=init)
    '''
    # models = [ (['fc', 500], ['fc', 1000], ['fc', 1000]) ]

    # model_name = 'mnist_adam_b50'
    # models = [ [ ['conv',(12,12,1,64),(2,2)],
    #              ['conv',(5,5,64,128),(2,2)],
    #              ['fc',200]],
    #            [ ['conv',(12,12,1,64),(2,2)],
    #              ['conv',(5,5,64,128),(2,2)],
    #              ['fc',500]],
    #            [ ['conv',(12,12,1,64),(2,2)],
    #              ['conv',(5,5,64,128),(2,2)],
    #              ['fc',200],
    #              ['fc',100]]
    # ]

    model_name = 'mnist_adam_b40'
    models = [ [ ['conv',(12,12,1,64),(2,2)],
                 ['conv',(5,5,64,128),(2,2)],
                 ['fc',200]],
               [ ['conv',(12,12,1,64),(4,4)],
                 ['conv',(3,3,64,128),(1,1)],
                 ['fc',500]],
    ]

    # dumb way to validate graphs so that you won't wake up to an error
    for idx, model in enumerate(models):
        name = 'test%d' % idx
        dbm = build_dcbm(model, name)

        tf.reset_default_graph()

    for idx, model in enumerate(models):
        name = model_name + '_%d' %idx
        dbm = build_dcbm(model, name)

        # if not os.path.exists(output_dir+name):
        #     os.mkdir(output_dir+name)
        if model[0][0]=='fc':
            train_rbm.train(dbm, train_xs_fc, lr, 500, batch_size, use_pcd, cd_k, 28, output_dir+name, init=init_fc, cutoff=0.6)
        else:
            train_rbm.train(dbm, train_xs, lr, 500, batch_size, use_pcd, cd_k, 28, output_dir+name, init=init, cutoff=0.6)

        tf.reset_default_graph()
