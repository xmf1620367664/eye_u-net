# _*_ coding:utf-8 _*_
import os
import numpy as np
import scipy.io as sci
from PIL import Image
#数据集划分
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import OneHotEncoder

mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

'''
    比较好的讲解:https://zhuanlan.zhihu.com/p/31428783
                http://www.sohu.com/a/270896638_633698
    
'''

class DATA():
    def __init__(self,
                 images_dir='./datasets/training/images',
                 labels_dir='./datasets/training/1st_manual',
                 test_dir='./datasets/test/images',
                 upload_dir='./datasets/test/predict',
                 categories=2):
        self.images_dir=images_dir
        self.labels_dir=labels_dir
        self.categories=categories
        self.test_dir=test_dir
        self.upload_dir=upload_dir
        #print(os.getcwd())
        self.train_images,self.test_images,self.train_labels,\
                    self.test_labels=self.get_one_hot_data()
        print('训练集样本:',self.train_images.shape,self.train_labels.shape)
        print('测试集样本:',self.test_images.shape,self.test_labels.shape)

    def trans_label(self):
        categories_list=[i for i in range(self.categories)]
        categories_array=np.array(categories_list).reshape([-1,1])
        one_hot_index=OneHotEncoder(sparse=True).fit_transform(categories_array).toarray()
        return one_hot_index

    #获取数据加数据集划分
    def get_one_hot_data(self):
        one_hot=self.trans_label()

        images_list=os.listdir(self.images_dir)
        labels_list=[i[:3]+'manual1.gif' for i in images_list]
        samples_len=len(images_list)

        images_array=[]
        labels_array=[]
        for i in range(samples_len):
            #images
            item_image_path=os.path.join(self.images_dir,images_list[i])
            item_image=Image.open(item_image_path)
            item_image=np.array(item_image)
            assert item_image.shape==(584,565,3)
            images_array.append(item_image)


            #labels
            new_label=np.zeros([584,565,self.categories],dtype=np.uint8)
            item_label_path=os.path.join(self.labels_dir,labels_list[i])
            item_label=Image.open(item_label_path)
            item_label_metric=np.array(item_label)
            for i in range(item_label_metric.shape[0]):
                for j in range(item_label_metric.shape[1]):
                    if item_label_metric[i,j]==255:
                        index=1
                    else:
                        index=0
                    new_label[i,j]=one_hot[index]
            labels_array.append(new_label)
        images_array=np.array(images_array)
        labels_array=np.array(labels_array)
        #划分
        # 训练集、测试集划分
        train_images, test_images, train_labels, test_labels = train_test_split(images_array, labels_array, test_size=0.1, random_state=1024)
        return train_images,test_images,train_labels,test_labels

    def get_testData(self):
        test_list=[str(i+1).zfill(2)+'_test.tif' for i in range(20)]
        images_array=[]
        for i in range(len(test_list)):
            item_path=os.path.join(self.test_dir,test_list[i])
            item_image=Image.open(item_path)
            item_image=np.array(item_image)
            assert item_image.shape == (584, 565, 3)
            images_array.append(item_image)
        images_array=np.array(images_array)
        return images_array

    #设置类别颜色
    def set_colour(self,number):
        colour_categories={0:[0,0,0],1:[255,255,255]}
        return colour_categories[number]

    #转化为rgb图
    def trans_colour(self,image):
        try:
            img_array=np.array(image).reshape([584,565,self.categories])
        except BaseException:
            print('Image Shape Error!')
        new_array=[]
        #lines=[]
        for i in range(img_array.shape[0]):
            #cows=[]
            for j in range(img_array.shape[1]):
                index=np.argmax(img_array[i][j])
                new_array.append(
                        self.set_colour(index))
        new_array=np.array(new_array).reshape([img_array.shape[0],img_array.shape[1],3])
        #print(new_array.shape)
        return new_array

    def trans_colour_2(self,image):
        try:
            img_array=np.array(image).reshape([584,565])
        except BaseException:
            print('Image Shape Error!')
        new_array = []
        # lines=[]
        for i in range(img_array.shape[0]):
            # cows=[]
            for j in range(img_array.shape[1]):
                #index = np.argmax(img_array[i][j])
                new_array.append(
                    self.set_colour(img_array[i][j]))
        new_array = np.array(new_array).reshape([img_array.shape[0], img_array.shape[1], 3])
        # print(new_array.shape)
        return new_array
    #数据展示
    def show_label(self,image):
        new_array=self.trans_colour(np.uint8(image))
        new_img=Image.fromarray(np.uint8(new_array))
        new_img.show()

    def show_image(self,image):
        img=Image.fromarray(np.uint8(image))
        img.show()

    def show_dataset(self,show_size=(6,6)):
        print('show_size', show_size)
        f, a = plt.subplots(show_size[0], show_size[1], figsize=(10, 10))
        plt.suptitle('数据集概要展示')
        # f.suptitle()
        for i in range(show_size[0]):
            # print('i',i)
            if i%2==0:
                for j in range(show_size[1]):
                    # print('j',j)
                    tmp_x = self.train_images[(i//2) * show_size[0] + j]#.reshape([32, 32, 3])
                    a[i][j].imshow(tmp_x)
                    a[i][j].axis('off')
            if i%2==1:
                for j in range(show_size[1]):
                    # print('j',j)
                    tmp_x = self.train_labels[((i-1)//2) * show_size[0] + j]#.reshape([32, 32, 3])
                    tmp_x=self.trans_colour(np.uint8(tmp_x))
                    a[i][j].imshow(tmp_x)
                    a[i][j].axis('off')
        plt.show()

    def show_result(self,images, labels, show_size=(4, 10)):
        print('show_size', show_size)
        f, a = plt.subplots(show_size[0], show_size[1], figsize=(10, 10))
        plt.suptitle('SHOW UPLOAD RESULT')
        # f.suptitle()
        for i in range(show_size[0]):
            # print('i',i)
            if i % 2 == 0:
                for j in range(show_size[1]):
                    # print('j',j)
                    tmp_x = images[i//2 * show_size[0] + j]  # .reshape([32, 32, 3])
                    try:
                        tmp_x = self.trans_colour(np.uint8(tmp_x))
                    except BaseException:
                        tmp_x = images[i // 2 * show_size[0] + j]
                    #tmp_x=self.trans_colour_2(np.uint8(tmp_x))
                    a[i][j].imshow(tmp_x)
                    a[i][j].axis('off')
            if i % 2 == 1:
                for j in range(show_size[1]):
                    # print('j',j)
                    tmp_x = labels[(i - 1)//2 * show_size[0] + j]  # .reshape([32, 32, 3])
                    tmp_x = self.trans_colour_2(np.uint8(tmp_x ))
                    a[i][j].imshow(tmp_x)
                    a[i][j].axis('off')
        plt.show()

    def save_result(self,test_labels,result,target_dir):
        if not os.path.exists(os.path.join(target_dir, 'labels')):
            os.mkdir(os.path.join(target_dir, 'labels'))
        if not os.path.exists(os.path.join(target_dir, 'result')):
            os.mkdir(os.path.join(target_dir, 'result'))
        for i in range(len(test_labels)):
            tmp_x=test_labels[i]
            tmp_x=self.trans_colour(np.uint8(tmp_x))
            target_path=os.path.join(target_dir,'labels',str(i)+'.png')
            img=Image.fromarray(np.uint8(tmp_x))
            img.save(target_path)
            try:
                tmp_y=result[i]
            except BaseException:
                break
            tmp_y=self.trans_colour_2(np.uint8(tmp_y))
            target_path=os.path.join(target_dir,'result',str(i)+'.png')
            img=Image.fromarray(np.uint8(tmp_y))
            img.save(target_path)
    def save_result_npy(self,test_labels,result,target_dir):
        min=len(result)
        test_labels=test_labels[:min]
        target_path=os.path.join(target_dir,'labels.npy')
        np.save(target_path,test_labels)

        target_path=os.path.join(target_dir,'result.npy')
        np.save(target_path,result)

    def upload_result(self,result):
        if not os.path.exists(self.upload_dir):
            os.mkdir(self.upload_dir)
        for i in range(len(result)):
            tmp_y=result[i]
            tmp_y=self.trans_colour_2(np.uint8(tmp_y))
            target_path=os.path.join(self.upload_dir,str(i+1)+'.png')
            img=Image.fromarray(np.uint8(tmp_y))
            img.save(target_path)

    #数据接口
    def get_dateset(self):
        return self.train_images,self.test_images,\
                    self.train_labels,self.test_labels

if __name__=='__main__':
    dt=DATA()
    #a,b,c,d=dt.get_one_hot_data()
    #dt.trans_colour(c[0])
    #dt.show_label(c[0])
    #dt.show_image(a[0])
    #dt.show_dataset()
    #a,b,c,d,e=dt.get_dateset()
    #print(c.shape)
    #dt.trans_label()
    print(dt.get_testData().shape)
