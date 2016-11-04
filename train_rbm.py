import numpy as np
import cPickle
import os
import time
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import dbm as dbm_class
import utils


def load_model(sess, model_path):
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print 'Model loaded from:', model_path


def train(rbm, train_xs, lr, num_epoch, batch_size, use_pcd, cd_k, output_dir, init=None):
    vis_shape = train_xs.shape[1:]    # shape of single image
    batch_shape = (batch_size,) + vis_shape
    num_batches = len(train_xs) / batch_size
    assert num_batches * batch_size == len(train_xs)

    # initialize input for evaluation
    num_samples = 100
    init_shape = tuple([num_samples] + rbm.vis_shape)
    init_random = np.random.uniform(0, 1, init_shape).astype(np.float32)
    init_evo_random =init_random[:10]
    init_evo = init

    # graph related definitions
    ph_vis = tf.placeholder(tf.float32, batch_shape, name='vis_input')
    ph_lr = tf.placeholder(tf.float32, (), name='lr')
    if use_pcd:
        persistent_vis_holder = tf.placeholder(tf.float32, batch_shape, name='pst_vis_holder')
        persistent_vis_value = np.random.uniform(size=batch_shape).astype(np.float32)
    else:
        persistent_vis_holder = None

    # Build the graph
    loss, cost, new_vis = rbm.get_loss_updates(ph_lr, ph_vis, persistent_vis_holder, cd_k)
    opt = tf.train.GradientDescentOptimizer(ph_lr)
    train_step = opt.minimize(cost)

    # start a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    with sess.as_default():
        train_writer = tf.train.SummaryWriter('./train', sess.graph)
        tf.initialize_all_variables().run()

        for i in range(num_epoch):
            t = time.time()
            np.random.shuffle(train_xs)
            loss_vals = np.zeros(num_batches)
            cost_vals = np.zeros(num_batches)
            for b in range(num_batches):
                batch_xs = train_xs[b * batch_size:(b+1) * batch_size]

                if use_pcd:
                    loss_vals[b], cost_vals[b], persistent_vis_value = sess.run(
                        [loss, train_step, new_vis],
                        feed_dict={ph_vis: batch_xs,
                                   ph_lr: lr,
                                   persistent_vis_holder: persistent_vis_value})
                else:
                    loss_vals[b], cost_vals[b] = sess.run(
                            [loss,train_step], feed_dict={ph_vis: batch_xs, ph_lr: lr })
            print 'Epoch', i+1
            print '\tTrain Loss:', loss_vals.mean()
            print '\tTrain Cost:', cost_vals.mean()
            print '\tTraining time:', time.time() - t
            if output_dir is not None and (i+1)%5==0:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                saver = tf.train.Saver()
                save_path = saver.save(
                    sess, os.path.join(output_dir, '%s-epoch%d.ckpt' % (rbm.name, i)))
                print '\tModel saved to:', save_path

                num_steps = 1000
                '''
                # Generate samples
                t = time.time()
                gen_samples = rbm.sample_from_rbm(num_steps, init_random)
                prob_imgs, sampled_imgs = sess.run(gen_samples)
                img_path = os.path.join(output_dir, 'epoch%d-random.png' % i)
                imgs = prob_imgs.reshape(num_samples, -1)
                utils.vis_images(imgs.T, 10, 10, (28,28), img_path)
                '''
                # plot evolution, this is slow
                num_stages = 10
                num_samples = 10
                steps_per_stage = num_steps/num_stages
                imgs = np.zeros([num_stages] + list(init_evo_random.shape))
                imgs[0] = init_evo_random
                gen_samples = rbm.sample_from_rbm(steps_per_stage, init_evo_random)
                for j in range(1, num_stages):
                    prob_imgs, sampled_imgs = sess.run(gen_samples)
                    imgs[j] = prob_imgs
                    gen_samples = rbm.sample_from_rbm(steps_per_stage, prob_imgs)
                img_path = os.path.join(output_dir, 'epoch%d-random-evo.png' % (i+1))
                utils.vis_images(imgs.reshape(num_samples*num_stages,-1).T, 10, 10, (28,28), img_path)

                if init is not None:
                    imgs = np.zeros([num_stages] +list(init_evo.shape))
                    imgs[0] = init_evo
                    gen_samples = rbm.sample_from_rbm(steps_per_stage, init_evo)
                    for j in range(1, num_stages):
                        prob_imgs, sampled_imgs = sess.run(gen_samples)
                        imgs[j] = prob_imgs
                        gen_samples = rbm.sample_from_rbm(steps_per_stage, prob_imgs)
                    img_path = os.path.join(output_dir, 'epoch%d-test.png' % (i+1))
                    utils.vis_images(imgs.reshape(num_samples*num_stages, -1).T, 10, 10, (28,28), img_path)

                print '\tVisualization time:', time.time() - t
