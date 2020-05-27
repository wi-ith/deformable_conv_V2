import tensorflow as tf
import tensorflow.contrib as tc


def DeformableConv2D(input,
                     output_dims,
                     kernel_size,
                     stride,
                     idx,
                     seperable=False,
                     activation_fn=None,
                     normalizer_fn=None,
                     normalizer_params=None,
                     biases_initializer=None
                     ):
    with tf.variable_scope('DeformableConv2D'+'_'+idx, reuse=tf.AUTO_REUSE):

        if seperable:
            offsets = tc.layers.separable_conv2d(input, None, 3, 1,
                                                 stride=1,
                                                 activation_fn=tf.nn.relu6,
                                                 normalizer_fn=None, normalizer_params=None,
                                                 rate=1,
                                                 scope='offset_conv_depthwise')

            offsets = tc.layers.conv2d(offsets, kernel_size * kernel_size * 2, 1, activation_fn=tf.nn.sigmoid,
                                       normalizer_fn=None, normalizer_params=None,
                                       scope='offset_conv_pointwise')

            modulation = tc.layers.separable_conv2d(input, None, 3, 1,
                                                    stride=1,
                                                    activation_fn=tf.nn.relu6,
                                                    normalizer_fn=None, normalizer_params=None,
                                                    rate=1,
                                                    scope='weights_conv_depthwise')

            modulation = tc.layers.conv2d(modulation, kernel_size * kernel_size, 1, activation_fn=tf.nn.sigmoid,
                                          normalizer_fn=None, normalizer_params=None,
                                          scope='weights_conv_pointwise')

            modulation = tf.sigmoid(modulation)

        else:
            offsets = tc.layers.conv2d(input,
                                       kernel_size*kernel_size*2,
                                       kernel_size,
                                       stride=stride,
                                       activation_fn=None,
                                       normalizer_fn=None,
                                       normalizer_params=None,
                                       scope='offset_conv')

            modulation = tc.layers.conv2d(input,
                                       kernel_size*kernel_size,
                                       kernel_size,
                                       stride=stride,
                                       activation_fn=tf.nn.sigmoid,
                                       normalizer_fn=None,
                                       normalizer_params=None,
                                       scope='weights_conv')
            modulation = tf.sigmoid(modulation)
