# _*_ coding:utf-8 _*_
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'  #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from tensorflow.contrib.layers.python.layers import batch_norm

class UNet_Model():
    def __init__(self,input,batch_size=1,channel1=3,channel2=64,
                    channel3=128,channel4=256,channel5=512,
                    channel6=1024,categories=2):
        self.input=input
        self.batch_size=batch_size
        self.channel1=channel1
        self.channel2=channel2
        self.channel3 = channel3
        self.channel4 = channel4
        self.channel5 = channel5
        self.channel6 = channel6
        self.out_channel=categories
        self.categories=categories

    def get_Variable(self,name,shape,dtype=tf.float32,initializer=tf.random_normal_initializer(0,0.01)):
        return tf.get_variable(name,shape,dtype,initializer)

    def model(self,conv_size=3):
        nw = batch_norm(self.input, is_training=True)
        with tf.variable_scope('cn1'):
            nw=tf.nn.conv2d(nw,filter=self.get_Variable('w1',[conv_size,conv_size,self.channel1,self.channel2]),
                            strides=[1,1,1,1],padding='SAME')
            nw=tf.nn.bias_add(nw,self.get_Variable('b1',[self.channel2]))
            nw=tf.nn.relu(nw)
            nw = batch_norm(nw, is_training=True)
            nw = tf.nn.conv2d(nw, filter=self.get_Variable('w2', [conv_size, conv_size, self.channel2, self.channel2]),
                              strides=[1, 1, 1, 1], padding='SAME')
            nw = tf.nn.bias_add(nw, self.get_Variable('b2', [self.channel2]))
            nw1 = tf.nn.relu(nw)

            nw=tf.nn.max_pool(nw1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
            print('cn1',nw.get_shape())
        with tf.variable_scope('cn2'):
            nw=tf.nn.conv2d(nw,filter=self.get_Variable('w1',[conv_size,conv_size,self.channel2,self.channel3]),
                            strides=[1,1,1,1],padding='SAME')
            nw=tf.nn.bias_add(nw,self.get_Variable('b1',[self.channel3]))
            nw=tf.nn.relu(nw)
            nw = batch_norm(nw, is_training=True)
            nw = tf.nn.conv2d(nw, filter=self.get_Variable('w2', [conv_size, conv_size, self.channel3, self.channel3]),
                              strides=[1, 1, 1, 1], padding='SAME')
            nw = tf.nn.bias_add(nw, self.get_Variable('b2', [self.channel3]))
            nw2 = tf.nn.relu(nw)

            nw = tf.nn.max_pool(nw2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print('cn2', nw.get_shape())
        with tf.variable_scope('cn3'):
            nw=tf.nn.conv2d(nw,filter=self.get_Variable('w1',[conv_size,conv_size,self.channel3,self.channel4]),
                            strides=[1,1,1,1],padding='SAME')
            nw=tf.nn.bias_add(nw,self.get_Variable('b1',[self.channel4]))
            nw = tf.nn.relu(nw)
            nw = batch_norm(nw, is_training=True)
            nw = tf.nn.conv2d(nw, filter=self.get_Variable('w2', [conv_size, conv_size, self.channel4, self.channel4]),
                              strides=[1, 1, 1, 1], padding='SAME')
            nw = tf.nn.bias_add(nw, self.get_Variable('b2', [self.channel4]))
            nw3 = tf.nn.relu(nw)

            nw = tf.nn.max_pool(nw3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print('cn3', nw.get_shape())
        with tf.variable_scope('cn4'):
            nw=tf.nn.conv2d(nw,filter=self.get_Variable('w1',[conv_size,conv_size,self.channel4,self.channel5]),
                            strides=[1,1,1,1],padding='SAME')
            nw=tf.nn.bias_add(nw,self.get_Variable('b1',[self.channel5]))
            nw = tf.nn.relu(nw)
            nw = batch_norm(nw, is_training=True)
            nw = tf.nn.conv2d(nw, filter=self.get_Variable('w2', [conv_size, conv_size, self.channel5, self.channel5]),
                              strides=[1, 1, 1, 1], padding='SAME')
            nw = tf.nn.bias_add(nw, self.get_Variable('b2', [self.channel5]))
            nw4 = tf.nn.relu(nw)

            nw = tf.nn.max_pool(nw4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print('cn4', nw.get_shape())
        with tf.variable_scope('cn5'):
            nw=tf.nn.conv2d(nw,filter=self.get_Variable('w1',[conv_size,conv_size,self.channel5,self.channel6]),
                            strides=[1,1,1,1],padding='SAME')
            nw=tf.nn.bias_add(nw,self.get_Variable('b1',[self.channel6]))
            nw = tf.nn.relu(nw)
            nw = batch_norm(nw, is_training=True)
            nw = tf.nn.conv2d(nw, filter=self.get_Variable('w2', [conv_size, conv_size, self.channel6, self.channel6]),
                              strides=[1, 1, 1, 1], padding='SAME')
            nw = tf.nn.bias_add(nw, self.get_Variable('b2', [self.channel6]))
            nw = tf.nn.relu(nw)
            #nw5 = tf.nn.max_pool(nw, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            print('cn5', nw.get_shape())

        with tf.variable_scope('cn4_T'):
            nw=tf.nn.conv2d_transpose(nw,filter=self.get_Variable('w1',shape=[conv_size,conv_size,self.channel5,self.channel6]),
                                      output_shape=[self.batch_size,73,71,self.channel5],strides=[1,2,2,1])
            nw=tf.nn.bias_add(nw,self.get_Variable('b1',[self.channel5]))
            nw = tf.concat([nw, nw4],axis=-1)
            nw=tf.nn.relu(nw)
            nw = batch_norm(nw, is_training=True)
            nw = tf.nn.conv2d(nw, filter=self.get_Variable('w2', [conv_size, conv_size, self.channel5*2, self.channel5]),
                              strides=[1, 1, 1, 1], padding='SAME')
            nw = tf.nn.bias_add(nw, self.get_Variable('b2', [self.channel5]))
            nw = tf.nn.relu(nw)
            nw = batch_norm(nw, is_training=True)
            nw = tf.nn.conv2d(nw, filter=self.get_Variable('w3', [conv_size, conv_size, self.channel5, self.channel5]),
                              strides=[1, 1, 1, 1], padding='SAME')
            nw = tf.nn.bias_add(nw, self.get_Variable('b3', [self.channel5]))
            nw = tf.nn.relu(nw)
            print('cn4_T', nw.get_shape())
        with tf.variable_scope('cn3_T'):
            nw=tf.nn.conv2d_transpose(nw,filter=self.get_Variable('w1',shape=[conv_size,conv_size,self.channel4,self.channel5]),
                                      output_shape=[self.batch_size,146,142,self.channel4],strides=[1,2,2,1])
            nw=tf.nn.bias_add(nw,self.get_Variable('b1',[self.channel4]))
            nw = tf.concat([nw, nw3],axis=-1)
            nw=tf.nn.relu(nw)
            nw = batch_norm(nw, is_training=True)
            nw = tf.nn.conv2d(nw, filter=self.get_Variable('w2', [conv_size, conv_size, self.channel4*2, self.channel4]),
                              strides=[1, 1, 1, 1], padding='SAME')
            nw = tf.nn.bias_add(nw, self.get_Variable('b2', [self.channel4]))
            nw = tf.nn.relu(nw)
            nw = batch_norm(nw, is_training=True)
            nw = tf.nn.conv2d(nw, filter=self.get_Variable('w3', [conv_size, conv_size, self.channel4, self.channel4]),
                              strides=[1, 1, 1, 1], padding='SAME')
            nw = tf.nn.bias_add(nw, self.get_Variable('b3', [self.channel4]))
            nw = tf.nn.relu(nw)
            print('cn3_T', nw.get_shape())
        with tf.variable_scope('cn2_T'):
            nw=tf.nn.conv2d_transpose(nw,filter=self.get_Variable('w1',shape=[conv_size,conv_size,self.channel3,self.channel4]),
                                      output_shape=[self.batch_size,292,283,self.channel3],strides=[1,2,2,1])
            nw=tf.nn.bias_add(nw,self.get_Variable('b1',[self.channel3]))
            nw = tf.concat([nw, nw2],axis=-1)
            nw=tf.nn.relu(nw)
            nw = batch_norm(nw, is_training=True)
            nw = tf.nn.conv2d(nw, filter=self.get_Variable('w2', [conv_size, conv_size, self.channel3*2, self.channel3]),
                              strides=[1, 1, 1, 1], padding='SAME')
            nw = tf.nn.bias_add(nw, self.get_Variable('b2', [self.channel3]))
            nw = tf.nn.relu(nw)
            nw = batch_norm(nw, is_training=True)
            nw = tf.nn.conv2d(nw, filter=self.get_Variable('w3', [conv_size, conv_size, self.channel3, self.channel3]),
                              strides=[1, 1, 1, 1], padding='SAME')
            nw = tf.nn.bias_add(nw, self.get_Variable('b3', [self.channel3]))
            nw = tf.nn.relu(nw)
            print('cn2_T', nw.get_shape())
        with tf.variable_scope('cn1_T'):
            nw=tf.nn.conv2d_transpose(nw,filter=self.get_Variable('w1',shape=[conv_size,conv_size,self.channel2,self.channel3]),
                                      output_shape=[self.batch_size,584,565,self.channel2],strides=[1,2,2,1])
            nw=tf.nn.bias_add(nw,self.get_Variable('b1',[self.channel2]))
            nw = tf.concat([nw, nw1],axis=-1)
            nw=tf.nn.relu(nw)
            nw = batch_norm(nw, is_training=True)
            nw = tf.nn.conv2d(nw,
                              filter=self.get_Variable('w2', [conv_size, conv_size, self.channel2 * 2, self.channel2]),
                              strides=[1, 1, 1, 1], padding='SAME')
            nw = tf.nn.bias_add(nw, self.get_Variable('b2', [self.channel2]))
            nw = tf.nn.relu(nw)
            nw = batch_norm(nw, is_training=True)
            nw = tf.nn.conv2d(nw, filter=self.get_Variable('w3', [conv_size, conv_size, self.channel2, self.channel2]),
                              strides=[1, 1, 1, 1], padding='SAME')
            nw = tf.nn.bias_add(nw, self.get_Variable('b3', [self.channel2]))
            nw = tf.nn.relu(nw)
            print('cn1_T', nw.get_shape())
        with tf.variable_scope('out'):
            nw = tf.nn.conv2d(nw, filter=self.get_Variable('w', [conv_size, conv_size, self.channel2, self.out_channel]),
                              strides=[1, 1, 1, 1], padding='SAME')
            nw = tf.nn.bias_add(nw, self.get_Variable('b', [self.out_channel]))
            nw=tf.nn.softmax(nw,axis=-1)
            print('out',nw.get_shape())
            return nw

if __name__=='__main__':
    batch_size=1
    X= tf.placeholder(tf.float32, [batch_size, 584,565,3], name='X')
    fcn=UNet_Model(X)
    fcn.model()
