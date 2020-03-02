from pylab import *
import os
import sys
import time


def calculate_iou( nb_classes,res_dir='./predict/unet/result', label_dir='./predict/unet/labels',
                   labels_path='./predict/unet/labels.npy',result_path='./predict/unet/result.npy'):
    conf_m = zeros((nb_classes, nb_classes), dtype=float)+0.001
    total = 0
    # mean_acc = 0.
    labels_npy=np.load(labels_path)
    result_npy=np.load(result_path)

    len_img=len(os.listdir(res_dir))
    for img_num in range(len_img):
        total += 1
        print('#%d '% ( total))
        #pred = img_to_array(Image.open(os.path.join(res_dir,str(img_num)+'.jpg'))).astype(int)
        #label = img_to_array(Image.open(os.path.join(label_dir,str(img_num)+'.jpg'))).astype(int)
        pred=np.uint8(result_npy[img_num])
        label=np.uint8(np.argmax(labels_npy[img_num],axis=-1))
        #flat_pred = np.ravel(pred)
        flat_pred=pred.flatten()
        #flat_label = np.ravel(label)
        flat_label=label.flatten()
        # acc = 0.
        for p, l in zip(flat_pred, flat_label):
            if l == 255:
                continue
            if l < nb_classes and p < nb_classes:
                conf_m[l, p] += 1
            else:
                print('Invalid entry encountered, skipping! Label: ', l,
                      ' Prediction: ', p, ' Img_num: ', img_num)
    I = np.diag(conf_m)
    U = np.sum(conf_m, axis=0)  +np.sum(conf_m, axis=1) - I#
    IOU = I/U
    meanIOU = np.mean(IOU)
    return conf_m, IOU, meanIOU


def evaluate( nb_classes=2,save_file='./predict/unet/index.txt'):
    start_time = time.time()
    duration = time.time() - start_time
    print('{}s used to make predictions.\n'.format(duration))

    start_time = time.time()
    conf_m, IOU, meanIOU = calculate_iou(nb_classes)
    print('IOU: ')
    print(IOU)
    print('meanIOU: %f' % meanIOU)
    print('pixel acc: %f' % (np.sum(np.diag(conf_m))/np.sum(conf_m)))
    duration = time.time() - start_time
    print('{}s used to calculate IOU.\n'.format(duration))
    with open(save_file,'w+',encoding='utf-8')as file:
        file.write('IOU: '+str(IOU)+'\n')
        file.write('meanIOU: %f' % meanIOU+'\n')
        file.write('pixel acc: %f' % (np.sum(np.diag(conf_m))/np.sum(conf_m))+'\n')
        file.write('{}s used to calculate IOU.\n'.format(duration))

if __name__ == '__main__':
    evaluate()