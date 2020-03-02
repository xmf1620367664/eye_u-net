# _*_ coding:utf-8 _*_
#======用于PyQt多次调用==========

import tensorflow as tf
from U_Net import UNet_Model
from PIL import Image
import numpy as np

class Pred():
    def __init__(self,batch_size=1,checkpoint_dir='./log/000'):
        self.batch_size=batch_size
        self.checkpoint_dir=checkpoint_dir
        # 定义输入占位符
        self.image = tf.placeholder(tf.float32, [self.batch_size, 584, 565, 3], name='image')
        self.unet = UNet_Model(self.image)
        self.out = self.unet.model()
        self.predict = tf.argmax(self.out, axis=-1)
        # Session()准备信息
        self.init = tf.global_variables_initializer()
        # 定义saver
        self.saver = tf.train.Saver(max_to_keep=2)
        self.sess=tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
        self.sess.run(self.init)
        if True:
            # 加载模型继续训练
            ckpt = tf.train.latest_checkpoint(self.checkpoint_dir)
            if ckpt:
                print("load model …………")
                self.saver.restore(self.sess, ckpt)
            else:
                pass

    #GUI样本预测
    def gui_predict(self,test_image):
        assert test_image.shape==(584,565,3)
        curr_image=np.reshape(test_image,[1,584,565,3])
        predict_=self.sess.run(self.predict,feed_dict={self.image:curr_image})
        return predict_

if __name__=='__main__':
    test_image=Image.open('/home/dream/PythonProjects/U_Net_eye/datasets/test/images/01_test.tif')
    test_image=np.array(test_image)
    pred=Pred()
    result=pred.gui_predict(test_image)
    #print(result.shape)
    result=np.reshape(result,[584,565])
    new_result=np.ones([584,565,3])
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if result[i][j]==0:
                new_result[i][j]=[0,0,0]
            else:
                new_result[i][j]=[255,255,255]
    result=Image.fromarray(np.uint8(new_result))
    result.show()