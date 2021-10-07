import tensorflow as tf
from tqdm import tqdm


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
    regression, classification = net(imgs)
    cls_value = smooth_l1_loss(targets0, regression)
    reg_value = focal_loss(targets1, classification)
    loss_value = reg_value + cls_value

    return loss_value, reg_value, cls_value

def fit_one_epoch(net, focal_loss, smooth_l1_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch):
    train_step  = get_train_step_fn()
    loss        = 0
    val_loss    = 0
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration>=epoch_step:
                break
            images, targets0, targets1 = batch[0], batch[1], batch[2]
            targets0 = tf.convert_to_tensor(targets0)
            targets1 = tf.convert_to_tensor(targets1)
            loss_value, _, _ = train_step(images, focal_loss, smooth_l1_loss, targets0, targets1, net, optimizer)
            loss = loss + loss_value

            pbar.set_postfix(**{'loss'  : loss.numpy() / (iteration + 1), 
                                'lr'    : optimizer._decayed_lr(tf.float32).numpy()})
            pbar.update(1)

    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration>=epoch_step_val:
                break
            images, targets0, targets1 = batch[0], batch[1], batch[2]
            targets0 = tf.convert_to_tensor(targets0)
            targets1 = tf.convert_to_tensor(targets1)
            loss_value, _, _ = val_step(images, focal_loss, smooth_l1_loss, targets0, targets1, net, optimizer)
            val_loss = val_loss + loss_value

            pbar.set_postfix(**{'val_loss': val_loss.numpy() / (iteration + 1)})
            pbar.update(1)

    logs = {'loss': loss.numpy() / epoch_step, 'val_loss': val_loss.numpy() / epoch_step_val}
    loss_history.on_epoch_end([], logs)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    net.save_weights('logs/ep%03d-loss%.3f-val_loss%.3f.h5' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))