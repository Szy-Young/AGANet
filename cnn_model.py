"""
Implement a multi-layer CNN model for action recognition
in tensorflow version.
"""

import tensorflow as tf
import os


class CNN_Model:
    def __init__(self,
                 dataset=None,
                 logdir='logs',
                 num_kp=25,
                 dim_kp=3,
                 seq_len=200,
                 num_class=51,
                 weight_decay=0,
                 base_lr=0.01,
                 lr_decay=0.5,
                 lr_decay_freq=80,
                 epoch=200,
                 epoch_size=100,
                 batch_size=256,
                 gpu_memory_fraction=1.0,
                 training=True):
        
        # Configure GPU usage
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
        tf.reset_default_graph()
        self.sess = tf.Session(config=config)
        
        # Set log dir
        self.log_dir = logdir
        self.writer = tf.summary.FileWriter(self.log_dir)
        
        # Net args        
        self.dataset = dataset
        self.num_kp = num_kp
        self.dim_kp = dim_kp
        self.seq_len = seq_len
        self.num_class = num_class
        
        # Training configurations
        self.training = training
        self.weight_decay = weight_decay
        self.base_lr = base_lr
        self.epoch = epoch
        self.epoch_size = epoch_size
        self.lr_decay = lr_decay
        self.lr_decay_freq = lr_decay_freq
        self.batch_size = batch_size   
        
        # Step learning rate policy
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.base_lr,
            self.global_step, self.lr_decay_freq*self.epoch_size, self.lr_decay,
            staircase=True)
        
        # Summaries
        self.summ_scalar_train_list = []
        self.summ_image_train_list = []
        self.summ_scalar_val_list = []
        self.summ_image_val_list = []
    
    
    def _build_ph(self):
        """ 
        Build Placeholder in tensorflow session.
        """
        with tf.name_scope('input'):
            # Input skeleton sequences
            self.skt = tf.placeholder(tf.float32, shape=[None, self.seq_len, self.num_kp, self.dim_kp], name='skeletons')
            # Output groundtruth category labels
            self.gt = tf.placeholder(tf.float32, shape=[None, self.seq_len, self.num_class], name='gt_label') 
            # Weight mask for counting loss
            self.mask = tf.placeholder(tf.float32, shape=[None, self.seq_len], name='weight_mask')
        print ("--Placeholder Built")
        
    
    def _build_train_op(self):
        """ 
        Build loss and optimizer.
        """
        with tf.name_scope('loss'):
            # Implement focal loss
            """
            pred = tf.nn.sigmoid(self.pred)
            pt = tf.where(tf.equal(self.gt, 1), pred, 1 - pred)
            alpha = 0.25
            gamma = 2
            epsilon = 1e-10
            loss = - alpha * tf.pow(1 - pt, gamma) * tf.log(pt + epsilon)
            loss = tf.reduce_sum(loss, axis=-1)
            """
            
            # Sigmoid CE category loss
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.gt, logits=self.pred, name='category_loss')
            loss = tf.reduce_sum(loss, axis=-1)
            loss = tf.multiply(loss, self.mask, name='masked_category_loss')
            
            # L2 regularization
            l2_reg = self.weight_decay * tf.reduce_mean(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables()], name='l2_weight_decay')
            
            self.loss = tf.reduce_mean(loss) + l2_reg
            self.summ_scalar_train_list.append(tf.summary.scalar('train_loss', self.loss))
            self.summ_scalar_val_list.append(tf.summary.scalar('lr', self.learning_rate))
            self.summ_scalar_val_list.append(tf.summary.scalar('val_loss', self.loss))
            self.summ_scalar_val_list.append(tf.summary.scalar('regularization', l2_reg))
        print ("--Loss & Scalar_summary Built")
        
        # Optimizer
        with tf.name_scope('optimizer'):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):   
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                #self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9, use_nesterov=True)
                self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step)
        print ("--Optimizer Built")
        
        
    def _build_monitor(self):
        """
        Visualize groundtruth and prediction results.
        """
        with tf.device('/cpu:0'):
            gt_vis = tf.expand_dims(tf.expand_dims(tf.transpose(self.gt[0]), 0), 3)
            pred_vis = tf.expand_dims(tf.expand_dims(tf.transpose(self.pred[0]), 0), 3)
            mask_vis = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.mask[0], 0), 1), 3)
            self.summ_image_train_list.append(tf.summary.image('train_gt_vis', gt_vis, max_outputs=1))
            self.summ_image_train_list.append(tf.summary.image('train_pred_vis', pred_vis, max_outputs=1))
            self.summ_image_train_list.append(tf.summary.image('train_mask_vis', mask_vis, max_outputs=1))
            self.summ_image_val_list.append(tf.summary.image('val_gt_vis', gt_vis, max_outputs=1))
            self.summ_image_val_list.append(tf.summary.image('val_pred_vis', pred_vis, max_outputs=1))
            self.summ_image_val_list.append(tf.summary.image('val_mask_vis', mask_vis, max_outputs=1))
        print ("--Image_summary Built")
        
        
    def _build_accuracy(self):
        """ 
        Computes accuracy tensor.
        """
        with tf.name_scope('accuracy'):
            gt_class = tf.argmax(self.gt, axis=-1)
            pred_class = tf.argmax(self.pred, axis=-1)
            accur = tf.reduce_mean(tf.cast(tf.equal(gt_class, pred_class), tf.float32))
            #threshold = 0.5
            #pred_class = tf.where((self.pred>threshold), tf.ones_like(self.pred), tf.zeros_like(self.pred))
            #accur = tf.reduce_mean(tf.cast(tf.equal(self.gt, pred_class), dtype=tf.float32))
            self.summ_scalar_train_list.append(tf.summary.scalar('train_accuracy', accur))
            self.summ_scalar_val_list.append(tf.summary.scalar('val_accuracy', accur))
        print ("--Accuracy_summary Built")
        
        
    def net(self, skt_input, name='CNN_action', norm_pos=False, norm_scale=False):
        """
        Network architecture.
        """
        with tf.variable_scope(name):
            # Normalize for invariance to position changes
            if norm_pos:
                skt = skt_input[:, :, :, :3]
                conf = skt_input[:, :, :, 3:]
                transposed_skt = tf.transpose(skt, [0, 1, 3, 2])
                sub_mean_w = tf.cast([0, 1/3, 1/3, 0, 0, 1/3, 0, 0, 0, 0], tf.float32, name='sub_mean_weight')
                sub_mean = tf.reduce_sum(tf.multiply(transposed_skt, sub_mean_w), axis=-1, keepdims=True)
                sub_mean = tf.tile(sub_mean, [1, 1, 1, self.num_kp])
                skt = skt - tf.transpose(sub_mean, [0, 1, 3, 2])
                skt_input = tf.concat([skt, conf], axis=-1)
            
            # Normalize for invariance to scale changes
            if norm_scale:
                skt = skt_input[:, :, :, :3]
                conf = skt_input[:, :, :, 3:]
                scale = tf.norm(skt[:, :, 1:2, :], axis=-1, keepdims=True)
                mean_scale = tf.reduce_mean(scale, axis=[1, 2, 3], keepdims=True)
                scale = tf.tile(scale, [1, 1, self.num_kp, self.dim_kp-1])
                mean_scale = tf.tile(mean_scale, [1, self.seq_len, self.num_kp, self.dim_kp-1])
                # Avoid division by 0
                scale = tf.where(tf.greater(scale, 0), scale, mean_scale)
                skt = (skt / scale) * mean_scale
                skt_input = tf.concat([skt, conf], axis=-1)
                            
            # Multi-layer CNN
            # Encode the spatial patterns
            x = tf.contrib.layers.batch_norm(skt_input, decay=0.9, epsilon=1e-5, is_training=self.training, scope='spatial_bn1')
            x = tf.layers.conv2d(x, 8, (3, 5), strides=(1, 2), padding='SAME', activation=tf.nn.relu, trainable=self.training, name='spatial_conv1')
            x = tf.contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-5, is_training=self.training, scope='spatial_bn2')
            x = tf.layers.conv2d(x, 32, (3, 5), padding='VALID', activation=tf.nn.relu, trainable=self.training, name='spatial_conv2')
            # Encode the temporal patterns
            x = tf.contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-5, is_training=self.training, scope='temporal_bn1')
            x = tf.layers.conv2d(x, 32, (5, 1), strides=(2, 1), padding='SAME', activation=tf.nn.relu, trainable=self.training, name='temporal_conv1')
            x = tf.contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-5, is_training=self.training, scope='temporal_bn2')
            x = tf.layers.conv2d(x, 64, (5, 1), strides=(2, 1), padding='SAME', activation=tf.nn.relu, trainable=self.training, name='temporal_conv2')
            x = tf.contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-5, is_training=self.training, scope='temporal_bn3')
            x = tf.layers.conv2d(x, 64, (5, 1), padding='SAME', activation=tf.nn.relu, trainable=self.training, name='temporal_conv3')
            x = tf.contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-5, is_training=self.training, scope='temporal_bn4')
            x = tf.layers.conv2d(x, 64, (5, 1), padding='SAME', activation=tf.nn.relu, trainable=self.training, name='temporal_conv4')
            # Upsample the temporal dimension
            x = tf.image.resize_images(x, size=(self.seq_len, 1))
            # Reduce the spatial dimension
            x = tf.squeeze(x, axis=2)
            
            # Classification layer
            if self.training:
                return tf.layers.dense(x, self.num_class, name='fc')
            else:
                return tf.layers.dense(x, self.num_class, activation=tf.nn.sigmoid, name='fc')
        
    
    def build_model(self):
        """ 
        Build model in tensorflow session.
        """
        self._build_ph()
        assert self.skt != None and self.gt != None
        
        self.pred = self.net(self.skt, norm_pos=False, norm_scale=False)
                
        if self.training:
            self._build_train_op()
            self._build_monitor()
            self._build_accuracy()
        
        # Initialize all variables
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if self.training:
            # Merge all summary
            self.summ_scalar_train = tf.summary.merge(self.summ_scalar_train_list)
            self.summ_image_train = tf.summary.merge(self.summ_image_train_list)
            self.summ_scalar_val = tf.summary.merge(self.summ_scalar_val_list)
            self.summ_image_val = tf.summary.merge(self.summ_image_val_list)
            self.writer.add_graph(self.sess.graph)
        print("--Model Built")
        
        
    def train(self):
        """
        Training process.
        """
        _epoch_count = 0
        _iter_count = 0
        
        # Set batch data generator
        self.train_generator = self.dataset.batch_generator(self.batch_size, sample_set='train')
        self.val_generator = self.dataset.batch_generator(self.batch_size, sample_set='val')
        
        # Perform training
        for n in range(self.epoch):
            for m in range(self.epoch_size):
                # Train step
                train_batch = next(self.train_generator)
                feed_dict = {self.skt: train_batch[0],
                             self.gt: train_batch[1],
                             self.mask: train_batch[2]}
                self.sess.run(self.train_step, feed_dict=feed_dict)
                
                if _iter_count % 10 == 0:
                    print ('--Epoch: %d, Iter: %d' %(_epoch_count, _iter_count))
                    # Record train loss and accuracy
                    self.writer.add_summary(
                        self.sess.run(self.summ_scalar_train, feed_dict=feed_dict), _iter_count)
                    # Visualize groundtruth and prediction in training set
                    self.writer.add_summary(
                        self.sess.run(self.summ_image_train, feed_dict=feed_dict), _iter_count)
                    # Val step
                    val_batch = next(self.val_generator)
                    feed_dict = {self.skt: val_batch[0],
                                 self.gt: val_batch[1],
                                 self.mask: val_batch[2]}
                    # Record val loss and accuracy
                    self.writer.add_summary(
                        self.sess.run(self.summ_scalar_val, feed_dict=feed_dict), _iter_count)
                    # Visualize groundtruth and prediction in validation set
                    self.writer.add_summary(
                        self.sess.run(self.summ_image_val, feed_dict=feed_dict), _iter_count)
                    del val_batch
                
                _iter_count += 1
                self.writer.flush()
                del train_batch
            
            _epoch_count += 1
            # Save model every epoch
            if self.log_dir is not None:
                self.saver.save(self.sess, os.path.join(self.log_dir, "model.ckpt"), n)
            

if __name__ == '__main__':
    # Test the model construction part
    cnn_model = CNN_Model(gpu_memory_fraction=0.33, training=True)
    cnn_model.build_model()
