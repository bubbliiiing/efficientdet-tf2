import os

import tensorflow as tf
from tqdm import tqdm


def get_train_step_fn(strategy):
    @tf.function
    def train_step(imgs, focal_loss, smooth_l1_loss, targets0, targets1, net, optimizer):
        with tf.GradientTape() as tape:
            # 计算loss
            regression, classification = net(imgs, training=True)
            reg_value   = smooth_l1_loss(targets0, regression)
            cls_value   = focal_loss(targets1, classification)
            loss_value  = reg_value + cls_value

        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss_value
    if strategy == None:
        return train_step
    else:
        #----------------------#
        #   多gpu训练
        #----------------------#
        @tf.function
        def distributed_train_step(imgs, focal_loss, smooth_l1_loss, targets0, targets1, net, optimizer):
            per_replica_losses = strategy.run(train_step, args=(imgs, focal_loss, smooth_l1_loss, targets0, targets1, net, optimizer,))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
        return distributed_train_step

#----------------------#
#   防止bug
#----------------------#
def get_val_step_fn(strategy):
    @tf.function
    def val_step(imgs, focal_loss, smooth_l1_loss, targets0, targets1, net, optimizer):
        regression, classification = net(imgs)
        cls_value   = smooth_l1_loss(targets0, regression)
        reg_value   = focal_loss(targets1, classification)
        loss_value  = reg_value + cls_value

        return loss_value
    if strategy == None:
        return val_step
    else:
        #----------------------#
        #   多gpu验证
        #----------------------#
        @tf.function
        def distributed_val_step(imgs, focal_loss, smooth_l1_loss, targets0, targets1, net, optimizer):
            per_replica_losses = strategy.run(val_step, args=(imgs, focal_loss, smooth_l1_loss, targets0, targets1, net, optimizer,))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses,
                                    axis=None)
        return distributed_val_step

def fit_one_epoch(net, focal_loss, smooth_l1_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, save_period, save_dir, strategy):
    train_step  = get_train_step_fn(strategy)
    val_step    = get_val_step_fn(strategy)
    
    loss        = 0
    val_loss    = 0
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration>=epoch_step:
                break
            images, targets0, targets1 = batch[0], batch[1], batch[2]
            loss_value  = train_step(images, focal_loss, smooth_l1_loss, targets0, targets1, net, optimizer)
            loss        = loss + loss_value

            pbar.set_postfix(**{'loss'  : loss.numpy() / (iteration + 1), 
                                'lr'    : optimizer._decayed_lr(tf.float32).numpy()})
            pbar.update(1)

    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration>=epoch_step_val:
                break
            images, targets0, targets1 = batch[0], batch[1], batch[2]
            loss_value  = val_step(images, focal_loss, smooth_l1_loss, targets0, targets1, net, optimizer)
            val_loss    = val_loss + loss_value

            pbar.set_postfix(**{'val_loss': val_loss.numpy() / (iteration + 1)})
            pbar.update(1)

    logs = {'loss': loss.numpy() / epoch_step, 'val_loss': val_loss.numpy() / epoch_step_val}
    loss_history.on_epoch_end([], logs)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        net.save_weights(os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.h5" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
