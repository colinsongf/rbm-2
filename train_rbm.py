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

def get_lr(lr, epoch, max_epoch):
    stage = float(epoch)/max_epoch
    if  stage < 0.1:
        return lr
    elif stage < 0.5:
        return lr*2
    elif stage < 0.9:
        return lr
    else: return lr/10
    #return lr

def train(rbm, train_xs, lr, num_epoch, batch_size, use_pcd, cd_k, img_size,output_dir, verbose=False, init=None, cutoffs=[]):
    vis_shape = train_xs.shape[1:]    # shape of single image
    batch_shape = (batch_size,) + vis_shape
    num_batches = len(train_xs) / batch_size
    assert num_batches * batch_size == len(train_xs)

    # initialize input for evaluation
    num_samples = 100
    init_shape = tuple([num_samples] + rbm.vis_shape)
    init_random = np.random.uniform(0, 1, init_shape).astype(np.float32)
    init_evo_random =init_random[:10]
    init_evo = init.astype(np.float32)

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
    # opt = tf.train.AdamOptimizer(learning_rate=ph_lr)
    if not verbose:
        train_step = opt.minimize(cost)
    else:
        grads = opt.compute_gradients(cost)
        apply_grads = opt.apply_gradients(grads)

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
                if not verbose:
                    if use_pcd:
                        loss_vals[b], _, persistent_vis_value = sess.run(
                            [loss, train_step, new_vis],
                            feed_dict={ph_vis: batch_xs,
                                    ph_lr: get_lr(lr, i, num_epoch),
                                    persistent_vis_holder: persistent_vis_value})
                    else:
                        loss_vals[b], _ = sess.run(
                                [loss,train_step], feed_dict={ph_vis: batch_xs, ph_lr: get_lr(lr, i, num_epoch) })
                else:
                    if use_pcd:
                        loss_vals[b], grads_val, persistent_vis_value = sess.run(
                            [loss, grads, new_vis],
                            feed_dict={ph_vis: batch_xs,
                                    ph_lr: get_lr(lr, i, num_epoch),
                                    persistent_vis_holder: persistent_vis_value})
                        _ = sess.run(apply_grads,
                                    feed_dict={ph_vis: batch_xs,
                                                ph_lr: get_lr(lr, i, num_epoch),
                                                persistent_vis_holder: persistent_vis_value})

                    else:
                        loss_vals[b], _ = sess.run(
                                [loss, grads], feed_dict={ph_vis: batch_xs, ph_lr: lr })
                        _ = sess.run(apply_grads,
                                    feed_dict={ph_vis: batch_xs, ph_lr: lr})
            print 'Epoch', i+1
            print '\tTrain Loss:', loss_vals.mean()
            print '\tTraining time:', time.time() - t
            # if verbose:
            #     print gradients and values...

            if output_dir is not None and (((i+1)%500==0) or ((i+1)%100==0 and (i+1)>4000)) :
            #if output_dir is not None and ( ((i+1)%20==0) or ((i+1)%10==0 and (i+1)>450)) :
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                saver = tf.train.Saver()
                save_path = saver.save(
                    sess, os.path.join(output_dir, '%s-epoch%d.ckpt' % (rbm.name, i+1)))
                print '\tModel saved to:', save_path

                num_steps = 1000

                # Generate samples
                num_samples = 100
                t = time.time()
                gen_samples = rbm.sample_from_rbm(num_steps, init_random)
                prob_imgs, sampled_imgs = sess.run(gen_samples)
                img_path = os.path.join(output_dir, 'epoch%d-random.png' % (i+1))
                imgs = prob_imgs.reshape(num_samples, -1)
                utils.vis_images(imgs.reshape(num_samples, -1).T, 10, 10, (img_size,img_size), img_path)

                for cutoff in cutoffs:
                    imgs_cutoff = np.zeros(imgs.shape)
                    imgs_cutoff[np.where(imgs>cutoff)] = 1.0
                    cutoff_path = os.path.join(output_dir, 'epoch%d-random-cf%.1f.png' % (i+1,cutoff))
                    utils.vis_images(imgs_cutoff.reshape(num_samples, -1).T, 10, 10, (img_size,img_size), cutoff_path)

                num_stages = 10
                num_samples = 10
                steps_per_stage = num_steps/num_stages

                # plot evolution, this is slow
                imgs = np.zeros([num_stages] + list(init_evo_random.shape))
                imgs[0] = init_evo_random
                gen_samples = rbm.sample_from_rbm(steps_per_stage, init_evo_random)
                for j in range(1, num_stages):
                    prob_imgs, sampled_imgs = sess.run(gen_samples)
                    imgs[j] = prob_imgs
                    gen_samples = rbm.sample_from_rbm(steps_per_stage, prob_imgs)
                img_path = os.path.join(output_dir, 'epoch%d-random-evo.png' % (i+1))
                utils.vis_images(imgs.reshape(num_samples*num_stages,-1).T, 10, 10, (img_size,img_size), img_path)

                if init is not None:
                    imgs = np.zeros([num_stages] +list(init_evo.shape))
                    imgs[0] = init_evo
                    gen_samples = rbm.sample_from_rbm(steps_per_stage, init_evo)
                    for j in range(1, num_stages):
                        prob_imgs, sampled_imgs = sess.run(gen_samples)
                        imgs[j] = prob_imgs
                        gen_samples = rbm.sample_from_rbm(steps_per_stage, prob_imgs)
                    img_path = os.path.join(output_dir, 'epoch%d-test.png' % (i+1))
                    utils.vis_images(imgs.reshape(num_samples*num_stages, -1).T, 10, 10, (img_size,img_size), img_path)

                print '\tVisualization time:', time.time() - t

'''
Same as train, but would same output images as np array for later rearrangement
'''
def train_demo(rbm, train_xs, lr, num_epoch, batch_size, use_pcd, cd_k, img_size,output_dir, verbose=False, init=None, cutoffs=[]):
    vis_shape = train_xs.shape[1:]    # shape of single image
    batch_shape = (batch_size,) + vis_shape
    num_batches = len(train_xs) / batch_size
    assert num_batches * batch_size == len(train_xs)

    # initialize input for evaluation
    num_samples = 100
    init_shape = tuple([num_samples] + rbm.vis_shape)
    init_random = np.random.uniform(0, 1, init_shape).astype(np.float32)
    init_evo_random =init_random[:10]
    init_evo = init.astype(np.float32)

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
    # opt = tf.train.GradientDescentOptimizer(ph_lr)
    opt = tf.train.AdamOptimizer(learning_rate=ph_lr)
    if not verbose:
        train_step = opt.minimize(cost)
    else:
        grads = opt.compute_gradients(cost)
        apply_grads = opt.apply_gradients(grads)

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
                if not verbose:
                    if use_pcd:
                        loss_vals[b], _, persistent_vis_value = sess.run(
                            [loss, train_step, new_vis],
                            feed_dict={ph_vis: batch_xs,
                                    ph_lr: get_lr(lr, i, num_epoch),
                                    persistent_vis_holder: persistent_vis_value})
                    else:
                        loss_vals[b], _ = sess.run(
                                [loss,train_step], feed_dict={ph_vis: batch_xs, ph_lr: get_lr(lr, i, num_epoch) })
                else:
                    if use_pcd:
                        loss_vals[b], grads_val, persistent_vis_value = sess.run(
                            [loss, grads, new_vis],
                            feed_dict={ph_vis: batch_xs,
                                    ph_lr: get_lr(lr, i, num_epoch),
                                    persistent_vis_holder: persistent_vis_value})
                        _ = sess.run(apply_grads,
                                    feed_dict={ph_vis: batch_xs,
                                                ph_lr: get_lr(lr, i, num_epoch),
                                                persistent_vis_holder: persistent_vis_value})

                    else:
                        loss_vals[b], _ = sess.run(
                                [loss, grads], feed_dict={ph_vis: batch_xs, ph_lr: lr })
                        _ = sess.run(apply_grads,
                                    feed_dict={ph_vis: batch_xs, ph_lr: lr})
            print 'Epoch', i+1
            print '\tTrain Loss:', loss_vals.mean()
            print '\tTraining time:', time.time() - t
            # if verbose:
            #     print gradients and values...

            if output_dir is not None and (((i+1)%500==0) or ((i+1)%300==0 and (i+1)>1500) or ((i+1)%100==0 and (i+1)>2500)):
            #if output_dir is not None and ( ((i+1)%20==0) or ((i+1)%10==0 and (i+1)>450)) :
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                saver = tf.train.Saver()
                save_path = saver.save(
                    sess, os.path.join(output_dir, '%s-epoch%d.ckpt' % (rbm.name, i+1)))
                print '\tModel saved to:', save_path

                num_steps = 1000

                # Generate samples
                num_samples = 100
                t = time.time()
                gen_samples = rbm.sample_from_rbm(num_steps, init_random)
                prob_imgs, sampled_imgs = sess.run(gen_samples)
                img_path = os.path.join(output_dir, 'epoch%d-random.png' % (i+1))
                imgs = prob_imgs.reshape(num_samples, -1)
                dump_path = os.path.join(output_dir, 'epoch%d-random.npz' % (i+1))
                np.savez_compressed(dump_path, imgs)
                utils.vis_images(imgs.reshape(num_samples, -1).T, 10, 10, (img_size,img_size), img_path)

                for cutoff in cutoffs:
                    imgs_cutoff = np.zeros(imgs.shape)
                    imgs_cutoff[np.where(imgs>cutoff)] = 1.0
                    cutoff_path = os.path.join(output_dir, 'epoch%d-random-cf%.1f.png' % (i+1,cutoff))
                    utils.vis_images(imgs_cutoff.reshape(num_samples, -1).T, 10, 10, (img_size,img_size), cutoff_path)
                    dump_path = os.path.join(output_dir, 'epoch%d-random-cf%.1f.npz' % (i+1,cutoff))
                    np.savez_compressed(dump_path, imgs_cutoff)

                num_stages = 10
                num_samples = 10
                steps_per_stage = num_steps/num_stages

                # plot evolution, this is slow
                imgs = np.zeros([num_stages] + list(init_evo_random.shape))
                imgs[0] = init_evo_random
                gen_samples = rbm.sample_from_rbm(steps_per_stage, init_evo_random)
                for j in range(1, num_stages):
                    prob_imgs, sampled_imgs = sess.run(gen_samples)
                    imgs[j] = prob_imgs
                    gen_samples = rbm.sample_from_rbm(steps_per_stage, prob_imgs)
                img_path = os.path.join(output_dir, 'epoch%d-random-evo.png' % (i+1))
                utils.vis_images(imgs.reshape(num_samples*num_stages,-1).T, 10, 10, (img_size,img_size), img_path)
                dump_path = os.path.join(output_dir, 'epoch%d-random-evo.npz' % (i+1))
                np.savez_compressed(dump_path, imgs)

                if init is not None:
                    imgs = np.zeros([num_stages] +list(init_evo.shape))
                    imgs[0] = init_evo
                    gen_samples = rbm.sample_from_rbm(steps_per_stage, init_evo)
                    for j in range(1, num_stages):
                        prob_imgs, sampled_imgs = sess.run(gen_samples)
                        imgs[j] = prob_imgs
                        gen_samples = rbm.sample_from_rbm(steps_per_stage, prob_imgs)
                    img_path = os.path.join(output_dir, 'epoch%d-test.png' % (i+1))
                    utils.vis_images(imgs.reshape(num_samples*num_stages, -1).T, 10, 10, (img_size,img_size), img_path)
                    dump_path = os.path.join(output_dir, 'epoch%d-test.npz' % (i+1))
                    np.savez_compressed(dump_path, imgs)

                print '\tVisualization time:', time.time() - t
