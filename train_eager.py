import time
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        TensorBoard)
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from nets.efficientdet import Efficientdet
from nets.efficientdet_training import Generator, focal, smooth_l1
from utils.anchors import get_anchors
from utils.utils import BBoxUtility, ModelCheckpoint


# 防止bug
def get_train_step_fn():
    @tf.function
    def train_step(imgs, focal_loss, smooth_l1_loss, targets0, targets1, net, optimizer):
        with tf.GradientTape() as tape:
            # 计算loss
            regression, classification = net(imgs, training=True)
            reg_value = smooth_l1_loss(targets0, regression)
            cls_value = focal_loss(targets1, classification)
            loss_value = reg_value + cls_value

        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss_value, reg_value, cls_value
    return train_step

@tf.function
def val_step(imgs, focal_loss, smooth_l1_loss, targets0, targets1, net, optimizer):
    # 计算loss
    regression, classification = net(imgs)
    cls_value = smooth_l1_loss(targets0, regression)
    reg_value = focal_loss(targets1, classification)
    loss_value = reg_value + cls_value

    return loss_value, reg_value, cls_value

def fit_one_epoch(net, focal_loss, smooth_l1_loss, optimizer, epoch, epoch_size, epoch_size_val, gen, genval, 
                Epoch, train_step=None):
    total_r_loss = 0
    total_c_loss = 0
    total_loss = 0
    
    val_loss = 0
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration>=epoch_size:
                break
            images, targets0, targets1 = batch[0], batch[1], batch[2]
            targets0 = tf.convert_to_tensor(targets0)
            targets1 = tf.convert_to_tensor(targets1)
            loss_value, reg_value, cls_value = train_step(images, focal_loss, smooth_l1_loss, targets0, targets1, net, optimizer)
            total_loss += loss_value
            total_c_loss += cls_value
            total_r_loss += reg_value

            pbar.set_postfix(**{'conf_loss'         : float(total_c_loss) / (iteration + 1), 
                                'regression_loss'   : float(total_r_loss) / (iteration + 1), 
                                'lr'                : optimizer._decayed_lr(tf.float32).numpy()})
            pbar.update(1)

            
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration>=epoch_size_val:
                break
            # 计算验证集loss
            images, targets0, targets1 = batch[0], batch[1], batch[2]
            targets0 = tf.convert_to_tensor(targets0)
            targets1 = tf.convert_to_tensor(targets1)

            loss_value, _, _ = val_step(images, focal_loss, smooth_l1_loss, targets0, targets1, net, optimizer)
            # 更新验证集loss
            val_loss = val_loss + loss_value

            pbar.set_postfix(**{'total_loss': float(val_loss)/ (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    net.save_weights('logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.h5'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
      
#---------------------------------------------------#
#   获得类和先验框
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

freeze_layers = [226, 328, 328, 373, 463, 463, 655, 802]
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408, 1536]

#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    #-------------------------------------------#
    #   训练前，请指定好phi和model_path
    #   二者所使用Efficientdet版本要相同
    #-------------------------------------------#
    phi = 0
    #----------------------------------------------------#
    #   获得图片路径和标签
    #----------------------------------------------------#
    annotation_path = '2007_train.txt'
    #----------------------------------------------------#
    #   classes的路径，非常重要
    #   训练前一定要修改classes_path，使其对应自己的数据集
    #----------------------------------------------------#
    classes_path = 'model_data/voc_classes.txt' 
    #------------------------------------------------------#
    #   一共有多少类和多少先验框
    #------------------------------------------------------#
    class_names = get_classes(classes_path)
    num_classes = len(class_names)  

    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #   训练自己的数据集时提示维度不匹配正常
    #   预测的东西都不一样了自然维度不匹配
    #------------------------------------------------------#
    model_path = "model_data/efficientdet-d0-voc.h5"

    #------------------------------------------------------#
    #   创建Efficientdet模型
    #------------------------------------------------------#
    model = Efficientdet(phi,num_classes=num_classes)
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    #-------------------------------#
    #   获得先验框
    #-------------------------------#
    priors = get_anchors(image_sizes[phi])
    bbox_util = BBoxUtility(num_classes, priors)

    #----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    #----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #------------------------------------------------------#
    for i in range(freeze_layers[phi]):
        model.layers[i].trainable = False

    if True:
        #--------------------------------------------#
        #   Batch_size不要太小，不然训练效果很差
        #--------------------------------------------#
        Batch_size = 8
        Lr = 1e-3
        Init_Epoch = 0
        Freeze_Epoch = 50

        generator = Generator(bbox_util, Batch_size, lines[:num_train], lines[num_train:],
                        (image_sizes[phi], image_sizes[phi]),num_classes)

        gen = partial(generator.generate, train = True, eager = True)
        gen = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32))
            
        gen_val = partial(generator.generate, train = False, eager = True)
        gen_val = tf.data.Dataset.from_generator(gen_val, (tf.float32, tf.float32, tf.float32))

        gen = gen.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)
        gen_val = gen_val.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)

        epoch_size = num_train//Batch_size
        epoch_size_val = num_val//Batch_size
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=Lr,
            decay_steps=epoch_size,
            decay_rate=0.95,
            staircase=True
        )

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, Batch_size))
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_one_epoch(model, focal(), smooth_l1(), optimizer, epoch, epoch_size, epoch_size_val, gen, gen_val, 
                        Freeze_Epoch, get_train_step_fn())

    for i in range(freeze_layers[phi]):
        model.layers[i].trainable = True

    if True:
        #--------------------------------------------#
        #   Batch_size不要太小，不然训练效果很差
        #--------------------------------------------#
        Batch_size = 4
        Lr = 5e-5
        Freeze_Epoch = 50
        Epoch = 100

        generator = Generator(bbox_util, Batch_size, lines[:num_train], lines[num_train:],
                        (image_sizes[phi], image_sizes[phi]),num_classes)

        gen = partial(generator.generate, train = True, eager = True)
        gen = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32))
            
        gen_val = partial(generator.generate, train = False, eager = True)
        gen_val = tf.data.Dataset.from_generator(gen_val, (tf.float32, tf.float32, tf.float32))

        gen = gen.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)
        gen_val = gen_val.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)


        epoch_size = num_train//Batch_size
        epoch_size_val = num_val//Batch_size
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=Lr,
            decay_steps=epoch_size,
            decay_rate=0.95,
            staircase=True
        )

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, Batch_size))
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        for epoch in range(Freeze_Epoch,Epoch):
            fit_one_epoch(model, focal(), smooth_l1(), optimizer, epoch, epoch_size, epoch_size_val, gen, gen_val, 
                        Epoch, get_train_step_fn())
