import numpy as np
import cPickle
import os
import time
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import cv2
import dbm as dbm_class
import utils


def load_model(sess, model_path):
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print 'Model loaded from:', model_path

def rescale_pixel(a):
    if(a!=0):
        a=0
    else:
        a=1
    return a
    
def load_img(img_path):
    im = cv2.imread(img_path,0)
    # Rescale and revert color
    rfunc = np.vectorize(rescale_pixel)
    return rfunc(im)
    
def test(rbm,model_path, output_dir, img_name_prefix, test_iters=50, num_steps=3000,noise_patterns=None):
    # Start a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # Load model
    load_model(sess,model_path)
    
    with sess.as_default():
        for i in xrange(test_iters):
            if noise_patterns!=None:
                num_noise = noise_patterns.shape[0]
                num_samples = num_noise * 10
                init = np.repeat(noise_patterns,10,axis=0)
                init = init.reshape(tuple([num_samples] + rbm.vis_shape))
                gen_samples = rbm.sample_from_rbm(num_steps, num_samples, init)
                prob_imgs, sampled_imgs = sess.run(gen_samples)
                img_path = os.path.join(output_dir, img_name_prefix+str(i)+'.png')    
                imgs = prob_imgs.reshape(num_samples, -1)
                utils.vis_samples(imgs, num_noise, 10, (28, 28), img_path)    
            
            else:
                num_samples = 64
                init_shape = tuple([num_samples] + rbm.vis_shape)
                init = np.random.normal(0, 1, init_shape).astype(np.float32)
                gen_samples = rbm.sample_from_rbm(num_steps, num_samples, init)
                prob_imgs, sampled_imgs = sess.run(gen_samples)
                img_path = os.path.join(output_dir, img_name_prefix+str(i)+'.png')    
                imgs = prob_imgs.reshape(num_samples, -1)
                utils.vis_samples(imgs, 8, 8, (28, 28), img_path)    
            plt.show()
