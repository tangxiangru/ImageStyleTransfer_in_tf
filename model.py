from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import vgg_model
STYLE_IMAGE = 'styles/' + STYLE + '.jpg'
CONTENT_IMAGE = 'content/' + CONTENT + '.jpg'
IMAGE_HEIGHT = 250
IMAGE_WIDTH = 333
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'

def create_content_loss(p, f):
    content_loss = tf.reduce_sum(tf.square(f-p))/(4*p.size)
    return content_loss

def gram_matrix(F, N, M):
    F = tf.reshape(F,(M,N))
    G = tf.matmul(tf.transpose(F),F)
    return G

def single_style_loss(a, g):

def create_style_loss(A, model):
    n_layers = len(STYLE_LAYERS)
    E = [single_style_loss(A[i], model[STYLE_LAYERS[i]]) for i in range(n_layers)]
    loss = 0
    for i in range(n_layers):
        loss += W[i]*E[i]
    return loss

def create_losses(model, input_image, content_image, style_image):
    with tf.variable_scope('loss') as scope:
        with tf.Session() as sess:
            sess.run(input_image.assign(content_image)) 
            p = sess.run(model[CONTENT_LAYER])
        content_loss=create_content_loss(p, model[CONTENT_LAYER])

        with tf.Session() as session:
            session.run(input_image.assign(style_image))
            A = session.run([model[layer_name] for layer_name in STYLE_LAYERS])
        style_loss = _create_style_loss(A, model)
        total_loss = 0.01*content_loss + 1*style_loss

    return content_loss, style_loss, total_loss

def create_summary(model):
    with tf.name_scope('summaries'):
        tf.summary.scalar('total_loss',model['total_loss'])
        tf.summary.scalar('content_loss',model['content_loss'])
        tf.summary.scalar('style_loss',model['style_loss'])
        return tf.summary.merge_all()

