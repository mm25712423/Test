# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library to train Inception using multiple GPUs with synchronous updates.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from datetime import datetime
import os.path
import re
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from Gan import image_processing
from Gan import image_models
from Gan.dataset import dataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/gan_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Learning rate decay factor.""")


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        num_batches_per_epoch = (dataset.num_examples_per_epoch() / FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay_factor,
                                        staircase=True)

        num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus
        images, labels = image_processing.distorted_inputs(dataset, num_preprocess_threads=num_preprocess_threads)

        num_classes = dataset.num_classes() + 1

        # Split the batch of images and labels for towers.
        images_splits = tf.split(axis=0, value=images)
        labels_splits = tf.split(axis=0, value=labels)

        z_placeholder = tf.placeholder(tf.float32, [None, FLAGS.batch_size], name='z_placeholder')

        with tf.device('/gpu:0'):
            with tf.name_scope('%s_' % ('Gan')) as scope:
                Gz = image_models.generator(z_placeholder, FLAGS.batch_size)
                Dx = image_models.discriminator(images)

                Dg = image_models.discriminator(Gz, reuse_variables=True)

                d_loss_real = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)))
                d_loss_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))

                g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))

        # Get the varaibles for different network
        tvars = tf.trainable_variables()

        d_vars = [var for var in tvars if 'd_' in var.name]
        g_vars = [var for var in tvars if 'g_' in var.name]

        # Train the discriminator
        d_trainer_fake = tf.train.AdamOptimizer(lr).minimize(d_loss_fake, var_list=d_vars)
        d_trainer_real = tf.train.AdamOptimizer(lr).minimize(d_loss_real, var_list=d_vars)

        # Train the generator
        g_trainer = tf.train.AdamOptimizer(lr).minimize(g_loss, var_list=g_vars)

        tf.get_variable_scope().reuse_variables()

        saver = tf.train.Saver()

        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))

        sess.run(tf.global_variables_initializer())

        # Pre-train discriminator
        for i in range(300):
            _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                                   {z_placeholder: FLAGS.batch_size})

            if (i % 100 == 0):
                print("dLossReal:", dLossReal, "dLossFake:", dLossFake)

        # Train generator and discriminator together
        for i in range(100000):
            # Train discriminator on both real and fake images
            _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                                   {z_placeholder: FLAGS.batch_size})

            # Train generator
            _ = sess.run(g_trainer)

            # if i % 10 == 0:
            #     # Update TensorBoard with summary statistics
            #     z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
            #     summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_image_batch})
            #     writer.add_summary(summary, i)

            if i % 1000 == 0:
                # Save the model every 1000 iteration
                save_path = saver.save(sess, "/tmp/model{}.ckpt".format(i))
                print("Model saved in file: %s" % save_path)

            if i % 100 == 0:
                # Every 100 iterations, show a generated image
                print("Iteration:", i, "at", datetime.datetime.now())
                generated_images = Gz = image_models.generator(z_placeholder, FLAGS.batch_size)
                images = sess.run(generated_images)
                plt.imshow(images[0].reshape([300, 300]))
                plt.savefig("/img/image{}.png".format(i))

                # Show discriminator's estimate
                im = images[0].reshape([1, 300, 300, 3])
                result = image_models.discriminator(im)
                estimate = sess.run(result)
                print("Estimate:", estimate)

if __main__ == '__main__':
    train()
