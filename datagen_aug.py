import os
import numpy as np
import random
from keras.utils import np_utils


class Data_Generator:
    def __init__(self,
                 skt_dir=None,
                 label_dir=None,
                 img_dir=None,
                 val_rate=0.1,
                 seq_len=100,
                 seq_step=30,
                 num_kp=10,
                 dim_kp=3,
                 num_class=11,
                 mask_label=None):
                 
        self.skt_dir = skt_dir
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.val_rate = val_rate
        
        self.seq_len = seq_len
        self.seq_step = seq_step
        self.num_kp = num_kp
        self.dim_kp = dim_kp
        self.num_class = num_class + 1
        self.mask_label = mask_label
    
    
    def load_labels(self, label_f):
        """
        Transform 'start_frame, end_frame, class' annotation to frame-wise class label.
        Generate frame-wise mask for counting loss according to class label.
        """        
       
        labels = []        
        # Read original annotations
        if os.path.exists(label_f):
            f = open(label_f, 'r')
            annotations = f.readlines()
        
            # Transform to frame-wise class label
            for ann in annotations:
                ann = ann.strip().split()
                if len(ann) == 0:
                    continue
                start_frame = int(ann[0])
                end_frame = int(ann[1])
                action_class = int(ann[2])
                labels.append((start_frame, end_frame, action_class))
            f.close()
        
        return labels
        
        
    def gaussian(self, scale, u, sigma, normalize=True):
        """
        Generate a 1-D Gaussian array. 
        """
        index = np.arange(scale, dtype=np.float32)
        array = np.exp(-((index - u)**2 / (2 * sigma**2))) 
        if normalize:
            array = array / np.max(array)
        return array       
        
        
    def load_set(self, data_dir):
        """
        Load all the clips into the dataset.
        """
        data_list = []
        # Traverse skeleton sequences of all videos in the set
        for skt_f in os.listdir(os.path.join(self.skt_dir, data_dir)):
            video = skt_f[:-4]
            skeletons = np.load(os.path.join(self.skt_dir, data_dir, skt_f), 'r')
            label_f = os.path.join(self.label_dir, data_dir, video+'.txt')
            labels = self.load_labels(label_f)
            assert skeletons.shape[1:] == (self.num_kp, self.dim_kp), 'Mismatched skeleton shape!'
            
            # Extract clips from the whole video 
            start_t = 0
            while start_t + self.seq_len < skeletons.shape[0]:
                # Clips around the time interval of actions are assigned as positive
                frame_label = self.time_iou(start_t, start_t+self.seq_len, labels)
                clip = {'skeleton': skeletons[start_t:(start_t+self.seq_len)],
                        'label': frame_label}
                data_list.append(clip)
                start_t += self.seq_step
        
        return data_list
        
        
    def time_iou(self, sw_start, sw_end, labels, threshold=0.8):
        """
        Calculate the IoU between the sliding window and action intervals. 
        Annotate an action if time IoU with the action is more than the threshold.
        """
        
        frame_label = np.zeros((self.seq_len), np.int32)
        for label in labels:
            label_start = label[0]
            label_end = label[1] + 1
            action_len = label_end - label_start
            action_class = label[2]
            # Return a positive sample
            if (min(sw_end, label_end) - max(sw_start, label_start)) > (threshold * action_len):
                start_t_in_seq = max(sw_start, label_start) - sw_start
                end_t_in_seq = min(sw_end, label_end) - sw_start
                frame_label[start_t_in_seq:end_t_in_seq] = action_class
            
        # Return a negative sample
        return frame_label
                
                
    def create_sets(self):
        """
        Shuffle the dataset and split into train and validation set.
        """
        
        self.val_list = self.load_set('val')
        self.train_list = self.load_set('train')
        print ('Dataset constructed')
        print ('--Training set: ', len(self.train_list), ' samples.')
        print ('--Validation set: ', len(self.val_list), ' samples.')
        
        
    def label_aug(self, scale=0.05):
        """
        Reload the training clips.
        Disturb the starting and ending time in labels.
        """
        
        self.train_list = []
        # Traverse skeleton sequences of all videos in the set
        for skt_f in os.listdir(os.path.join(self.skt_dir, 'train')):
            video = skt_f[:-4]
            skeletons = np.load(os.path.join(self.skt_dir, 'train', skt_f), 'r')
            label_f = os.path.join(self.label_dir, 'train', video+'.txt')
            labels = self.load_labels(label_f)
            assert skeletons.shape[1:] == (self.num_kp, self.dim_kp), 'Mismatched skeleton shape!'
            
            # Disturb the starting and ending time in labels
            auged_labels = []
            for label in labels:
                label_start = label[0]
                label_end = label[1] + 1
                action_len = label_end - label_start
                action_class = label[2]
                start_inc = int((np.random.uniform() * (2*scale) - scale) * action_len)
                auged_start = max(0, label_start+start_inc)
                end_inc = int((np.random.uniform() * (2*scale) - scale) * action_len)
                auged_end = min(skeletons.shape[0], label_end+end_inc)
                auged_label = (auged_start, auged_end, action_class)
                auged_labels.append(auged_label)
            
            # Extract clips from the whole video 
            start_t = 0
            while start_t + self.seq_len < skeletons.shape[0]:
                # Clips around the time interval of actions are assigned as positive
                frame_label = self.time_iou(start_t, start_t+self.seq_len, auged_labels)
                clip = {'skeleton': skeletons[start_t:(start_t+self.seq_len)],
                        'label': frame_label}
                self.train_list.append(clip)
                start_t += self.seq_step
        
        print ('Reload training set and disturb labels ...')
        return self.train_list
    
        
    def spatial_aug(self, skt, aug_seq_p=1, aug_frame_p=0, aug_joint_p=0):
    #def spatial_aug(self, skt, aug_seq_p=0.8, aug_frame_p=0.7, aug_joint_p=1.0):
        """
        Perform spatial data augmentation on a sequence from training set.
        Input:
            skt: The fixed length skeleton sequence, a numpy array of [T, num_kp, dim_kp].
            aug_seq_p: The probability to perform viewpoint disturbance on the whole sequence.
            aug_frame_p: The probability to perform viewpoint disturbance on each frame in the sequence.
            aug_joint_p: The probability to perform random noise disturbance on the coordinates of a single
                joint in a frame.
        """     
        
        skt = np.array(skt)        
        # Viewpoint disturbance on the whole sequence
        if np.random.binomial(1, aug_seq_p):
            skt = self.viewpoint_aug(skt, angle=(5, 5, 5), ss=0)
        
        """
        # Viewpoint disturbance on separate frames
        for t in range(skt.shape[0]):
            if np.random.binomial(1, aug_frame_p):
                print ('Aug Frame')
                skt[t] = self.viewpoint_aug(skt[t], angle=(5, 5, 5), ss=0.1)
        """
                             
        # Random noise on joints in separate frames
        if np.random.binomial(1, aug_joint_p):
            skt += np.random.normal(scale=0.02*np.absolute(skt))
        """
        for t in range(skt.shape[0]):
            for k in range(skt.shape[1]):
                if np.random.binomial(1, aug_joint_p):
                    skt[t, k] += np.random.normal(scale=0.02*np.absolute(skt[t, k]))
        """
                    
        return skt
                            
                  
    def viewpoint_aug(self, skt, angle=(90, 90, 90), ss=0.5):
        """
        Perform viewpoint disturbance for data augmentation on a whole skeleton
        sequence or a single skeleton frame.
        """
        
        # Random rotation matrix
        agx = (np.random.uniform() * (2*angle[0]) - angle[0]) * np.pi/180
        agy = (np.random.uniform() * (2*angle[1]) - angle[1]) * np.pi/180
        agz = (np.random.uniform() * (2*angle[2]) - angle[2]) * np.pi/180
        Rx = np.array([[1, 0, 0], [0, np.cos(agx), -np.sin(agx)], [0, np.sin(agx), np.cos(agx)]], np.float32)
        Ry = np.array([[np.cos(agy), 0, np.sin(agy)], [0, 1, 0], [-np.sin(agy), 0, np.cos(agy)]], np.float32)
        Rz = np.array([[np.cos(agz), -np.sin(agz), 0], [np.sin(agz), np.cos(agz), 0], [0, 0, 1]], np.float32)
        # Random scale matrix
        s = np.random.uniform() * (2*ss) - ss + 1
        sd = np.random.uniform() * (2*ss) - ss + 1
        Ss = np.array([[1, 0, 0], [0, 1, 0], [0, 0, sd]], np.float32)
        transform = np.matmul(np.matmul(np.matmul(Rz, Ry), Rx), Ss)
        
        return np.dot(skt, transform)
                    
        
    def batch_generator(self, batch_size=128, sample_set='train'):
        """
        Generate batch-wise data.
        """
        if sample_set == 'train':
            data_set = self.train_list
        elif sample_set == 'val':
            data_set = self.val_list
        else:
            raise Exception("Invalid sample set!")
            
        # Record the traverse of data set
        sample_idx = -1
        
        while True:
            # Construct batch data containers
            batch_skt = np.zeros((batch_size, self.seq_len, self.num_kp, self.dim_kp), np.float32)
            batch_gt = np.zeros((batch_size, self.seq_len, self.num_class-1), np.float32)
            batch_mask = np.ones((batch_size, self.seq_len), np.float32)
            
            i = 0
            while i < batch_size:
                sample_idx = (sample_idx + 1) % len(data_set)
                if sample_idx == 0 and sample_set == 'train':
                    data_set = self.label_aug()
                    random.shuffle(data_set)
                
                # Load skeleton and label
                skt = data_set[sample_idx]['skeleton']
                # Data augmentation
                if sample_set == 'train':
                    skt = np.concatenate((self.spatial_aug(skt[:, :, :3]), skt[:, :, 3:]), axis=-1)
                batch_skt[i] = skt
                label = data_set[sample_idx]['label']
                batch_gt[i] = np_utils.to_categorical(label, self.num_class)[:, 1:]
                
                i += 1
                
            yield batch_skt, batch_gt, batch_mask
                
