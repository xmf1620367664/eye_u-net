# _*_ coding:utf-8 _*_
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from data_yield import DATA
from U_Net import UNet_Model
import numpy as np
import os
import tensorflow as tf

class train():
    def __init__(self,batch_size=1,learning_rate=0.0001,categories=2,training_epochs=40,
                 print_num=1,save_num=4,checkpoint_dir='./log/000',save_path='./predict/unet'):
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.categories=categories
        self.training_epochs=training_epochs
        self.print_num=print_num
        self.save_num=save_num
        self.checkpoint_dir=checkpoint_dir
        self.save_path=save_path
        self.dt=DATA()

    def train(self):
        if os.path.exists(self.checkpoint_dir):
            pass
        else:
            os.mkdir(self.checkpoint_dir)

        train_image,test_images,train_labels,test_labels=self.dt.get_dateset()
        #dt.show_dataset()
        # 定义输入占位符
        image = tf.placeholder(tf.float32, [self.batch_size, 584,565,3], name='image')
        label = tf.placeholder(tf.float32, [self.batch_size, 584,565,self.categories], name='label')
        # 迭代次数，不可训练
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # 获取模型输出结果
        unet=UNet_Model(image)
        out=unet.model()
        #predict=tf.squeeze(out,[-1])
        predict=tf.argmax(out,axis=-1)

        # 获取softmax之后得的分类概率值
        # 计算交叉熵损失函数
        loss = tf.reduce_mean(-tf.reduce_sum(label * tf.log(out), axis=-1))
        # 使用梯度下降求解，最小化误差
        # train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
        train = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.99).minimize(loss)
        # Session()准备信息
        init = tf.global_variables_initializer()
        batch_num = int(train_image.shape[0] / self.batch_size)
        test_batch_num = test_images.shape[0] // self.batch_size
        # 定义saver
        saver = tf.train.Saver(max_to_keep=2)
        # 启动Session
        with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
            sess.run(init)
            if True:
                # 加载模型继续训练
                ckpt = tf.train.latest_checkpoint(self.checkpoint_dir)
                if ckpt:
                    print("load model …………")
                    saver.restore(sess, ckpt)
                else:
                    pass
            for epoch in range(sess.run(global_step), self.training_epochs):
                # 定义平均损失值
                average_loss = 0
                # 定义准确率
                train_correct_rate = 0
                # 定义随机索引
                index = np.random.permutation(train_image.shape[0])
                # 迭代batch_size
                for i in range(batch_num):
                    # 当前batch_size索引
                    train_index = index[i * self.batch_size:(i + 1) * self.batch_size]
                    # 获取batch数据
                    batch_x, batch_y = train_image[train_index], train_labels[train_index]
                    # 训练模型
                    sess.run(train, feed_dict={image: batch_x, label: batch_y})
                    # 计算平均损失
                    average_loss += sess.run(loss, feed_dict={image: batch_x, label: batch_y}) / batch_num
                    # 计算训练集正确率
                    #correct_rate_ = sess.run(correct_rate, feed_dict={image: batch_x, label: batch_y})
                    #train_correct_rate += correct_rate_ / batch_num
                # 打印信息
                if (epoch + 1) % self.print_num == 0:
                    print("训练集：\n epoch:{},loss:{}".format(epoch + 1, average_loss))
                    test_loss = 0
                    test_correct_rate = 0
                    for test_epoch in range(test_batch_num):
                        batch_x, batch_y = test_images[test_epoch * self.batch_size:(test_epoch + 1) * self.batch_size],\
                                           test_labels[test_epoch * self.batch_size:(test_epoch + 1) * self.batch_size]
                        test_loss_ = sess.run(loss,feed_dict={image: batch_x, label: batch_y})
                        test_loss += test_loss_ / test_batch_num
                        #test_correct_rate += test_correct_rate_ / test_batch_num
                    print("测试集：\n loss:{}".format(test_loss))
                # global_step累加
                sess.run(tf.assign(global_step, epoch + 1))
                # 保存模型
                if epoch % self.save_num == 0:
                    print('save model …………')
                    saver.save(sess, os.path.join(self.checkpoint_dir, 'model.ckpt'), global_step=global_step)
            saver.save(sess, os.path.join(self.checkpoint_dir, 'model.ckpt'), global_step=global_step)
            print("训练完成")

    #测试集预测
    def predict(self):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        dt = DATA()
        train_image, test_images, train_labels, test_labels = dt.get_dateset()
        # dt.show_dataset()
        # 定义输入占位符
        image = tf.placeholder(tf.float32, [self.batch_size, 584, 565, 3], name='image')
        label = tf.placeholder(tf.float32, [self.batch_size, 584, 565, self.categories], name='label')
        # 迭代次数，不可训练
        global_step = tf.Variable(0, name='global_step', trainable=False)

        #fcn = FCN_Model(image)
        # 获取模型输出结果
        #out = fcn.model()
        unet = UNet_Model(image)
        out = unet.model()
        # predict=tf.squeeze(out,[-1])
        # predict=tf.reshape(out,[batch_size,256,256,34])
        predict = tf.argmax(out, axis=-1)

        # Session()准备信息
        init = tf.global_variables_initializer()
        batch_num = int(train_image.shape[0] / self.batch_size)
        test_batch_num = test_images.shape[0] // self.batch_size
        # 定义saver
        saver = tf.train.Saver(max_to_keep=2)
        # 启动Session
        with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
            sess.run(init)
            if True:
                # 加载模型继续训练
                ckpt = tf.train.latest_checkpoint(self.checkpoint_dir)
                if ckpt:
                    print("load model …………")
                    saver.restore(sess, ckpt)
                else:
                    pass

            result=[]
            for test_epoch in range(test_batch_num):
                batch_x, batch_y = test_images[test_epoch * self.batch_size:(test_epoch + 1) * self.batch_size],\
                                   test_labels[test_epoch * self.batch_size:(test_epoch + 1) * self.batch_size]
                predict_=sess.run(predict,feed_dict={image:batch_x})
                for i in range(len(predict_)):
                    result.append(predict_[i])
            result=np.array(result)
            dt.show_result(test_labels,result)
            dt.save_result(test_labels,result,target_dir=self.save_path)
            dt.save_result_npy(test_labels,result,target_dir=self.save_path)

    #上传样本预测
    def upload_predict(self):
        test_images = self.dt.get_testData()
        # dt.show_dataset()
        # 定义输入占位符
        image = tf.placeholder(tf.float32, [self.batch_size, 584, 565, 3], name='image')

        unet = UNet_Model(image)
        out = unet.model()
        predict = tf.argmax(out, axis=-1)

        # Session()准备信息
        init = tf.global_variables_initializer()
        test_batch_num = test_images.shape[0] // self.batch_size
        # 定义saver
        saver = tf.train.Saver(max_to_keep=2)
        # 启动Session
        with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
            sess.run(init)
            if True:
                # 加载模型继续训练
                ckpt = tf.train.latest_checkpoint(self.checkpoint_dir)
                if ckpt:
                    print("load model …………")
                    saver.restore(sess, ckpt)
                else:
                    pass

            result=[]
            for test_epoch in range(test_batch_num):
                batch_x = test_images[test_epoch * self.batch_size:(test_epoch + 1) * self.batch_size]

                predict_=sess.run(predict,feed_dict={image:batch_x})
                for i in range(len(predict_)):
                    result.append(predict_[i])
            result=np.array(result)
            #self.dt.upload_result(result)
            self.dt.show_result(test_images,result)
            #dt.save_result(test_labels,result,target_dir=self.save_path)
            #dt.save_result_npy(test_labels,result,target_dir=self.save_path)

if __name__=='__main__':
    tr=train()
    #tr.train()
    #tr.predict()
    tr.upload_predict()
