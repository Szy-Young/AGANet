"""
Test trained model on specific skeleton sequences.
"""

#from cnn_model import CNN_Model
from cnn_model_att import CNN_Model
import os
import numpy as np
from matplotlib import pyplot as plt
import time
    

def visualize_predict_frame(skt_seq, labels, model, seq_len=200, seq_step=40, first_step=120):
    """
    Calculate frame-wise classification scores on a skeleton sequence.
    Visualize the groundtruth labels and prediction results in each sliding window.
    Input:
        skt_seq: 3D(+confidence) skeleton sequence, a [T, num_kp, dim_kp] numpy array.
        labels: Annotated action instances, a list [(start_time, end_time, action_class), ...].
    Output:
        final_preds: Frame-wise classification scores, a [T, num_class] numpy array.
    """
        
    # Average the prediction scores from several sliding windows
    sw_preds = []
    final_preds = np.zeros((skt_seq.shape[0], model.num_class), np.float32)
    
    threshold = 0.8
    curr = 0
    while skt_seq.shape[0] > seq_len:
        batch_skt = skt_seq[:seq_len]
        batch_skt = np.expand_dims(batch_skt, axis=0)
        # Predict frame-wise classification scores with deep CNN model
        batch_preds = model.sess.run(model.pred, feed_dict={model.skt: batch_skt})[0]
        skt_seq = skt_seq[seq_step:]
        # Generate groundtruth labels on the sliding window
        gt_labels = np.zeros((seq_len, model.num_class), np.float32)
        for label in labels:
            start_time = label[0]
            end_time = label[1] + 1
            action_len = end_time - start_time
            action_class = label[2] - 1
            if (min(curr+seq_len, end_time) - max(curr, start_time)) > threshold * action_len:
                start_t_in_sw = max(start_time, curr) - curr
                end_t_in_sw = min(end_time, curr+seq_len) - curr
                action_len = end_t_in_sw - start_t_in_sw + 1
                gt_labels[start_t_in_sw:end_t_in_sw, action_class] = 1
        # Visualize groundtruth labels and prediction scores
        time = np.arange(0, seq_len)
        for c in range(model.num_class):
            plt.subplot(model.num_class, 1, c+1)
            plt.xlim(0, seq_len)
            plt.ylim(0, 1)
            plt.plot(time, gt_labels[:, c], 'r-')
            plt.plot(time, batch_preds[:, c], 'b-')
        plt.show()
        curr += seq_step

    
def predict_frame(skt_seq, model, seq_len=200, seq_step=40, first_step=120):
    """
    Calculate frame-wise classification scores on a skeleton sequence.
    Input:
        skt_seq: 3D(+confidence) skeleton sequence, a [T, num_kp, dim_kp] numpy array.
    Output:
        final_preds: Frame-wise classification scores, a [T, num_class] numpy array.
    """
        
    final_preds = np.zeros((skt_seq.shape[0], model.num_class), np.float32)
    
    curr = 0
    while skt_seq.shape[0] > seq_len:
        batch_skt = skt_seq[:seq_len]
        batch_skt = np.expand_dims(batch_skt, axis=0)
        # Predict frame-wise classification scores with deep CNN model
        batch_preds = model.sess.run(model.pred, feed_dict={model.skt: batch_skt})[0]
        if skt_seq.shape[0] == final_preds.shape[0]:
            # Output scores on First Step frames at the start
            final_preds[:first_step] = batch_preds[:first_step]
            curr += first_step
        else:
            final_preds[curr:(curr+seq_step)] = batch_preds[(first_step-seq_step):first_step]
            curr += seq_step
        skt_seq = skt_seq[seq_step:]
    
    # Process left data less than a batch
    if (skt_seq.shape[0] + seq_step) > seq_len:
        pad_skt = np.zeros((seq_len, model.num_kp, model.dim_kp), np.float32)
        pad_skt[:(skt_seq.shape[0])] = skt_seq
        batch_skt = np.expand_dims(pad_skt, axis=0)
        batch_preds = model.sess.run(model.pred, feed_dict={model.skt: batch_skt})[0]
        final_preds[curr:] = batch_preds[:(final_preds.shape[0]-curr)]
    
    return final_preds
    
    
def predict_frame_sw(skt_seq, model, seq_len=200, seq_step=40, first_step=120):
    """
    Calculate frame-wise classification scores on a skeleton sequence.
    Input:
        skt_seq: 3D(+confidence) skeleton sequence, a [T, num_kp, dim_kp] numpy array.
    Output:
        final_preds: Frame-wise classification scores, a [T, num_class] numpy array.
    """
        
    # Average the prediction scores from several sliding windows
    sw_preds = []
    final_preds = np.zeros((skt_seq.shape[0], model.num_class), np.float32)
    
    curr = 0
    while skt_seq.shape[0] > seq_len:
        batch_skt = skt_seq[:seq_len]
        batch_skt = np.expand_dims(batch_skt, axis=0)
        # Predict frame-wise classification scores with deep CNN model
        batch_preds = model.sess.run(model.pred, feed_dict={model.skt: batch_skt})[0]
        skt_seq = skt_seq[seq_step:]
        # Accumulate predictions from several sliding windows and average them
        if len(sw_preds) == 0:
            # Output scores on First Step frames at the start
            sw_preds.append(batch_preds[first_step:])
            final_preds[:first_step] = batch_preds[:first_step]
            curr += first_step
        else:
            # Average and output scores on Seq Step frames continuously
            sw_preds.append(batch_preds[(first_step-seq_step):])
            avg_preds = np.stack([pred[:seq_step] for pred in sw_preds], axis=0)
            avg_preds = np.mean(avg_preds, axis=0)
            final_preds[curr:(curr+seq_step)] = avg_preds
            curr += seq_step
            sw_preds = [pred[seq_step:] for pred in sw_preds]
            # Remove useless history predictions
            if sw_preds[0].shape[0] == 0:
                sw_preds = sw_preds[1:]
    
    # Process left data less than a batch
    if (skt_seq.shape[0] + seq_step) > seq_len:
        pad_skt = np.zeros((seq_len, model.num_kp, model.dim_kp), np.float32)
        pad_skt[:(skt_seq.shape[0])] = skt_seq
        batch_skt = np.expand_dims(pad_skt, axis=0)
        batch_preds = model.sess.run(model.pred, feed_dict={model.skt: batch_skt})[0]
        sw_preds.append(batch_preds[(first_step-seq_step):])
    # Average left sliding window predictions
    while len(sw_preds) > 1:
        avg_preds = np.stack([pred[:seq_step] for pred in sw_preds], axis=0)
        avg_preds = np.mean(avg_preds, axis=0)
        final_preds[curr:(curr+seq_step)] = avg_preds
        curr += seq_step
        sw_preds = [pred[seq_step:] for pred in sw_preds]
        if sw_preds[0].shape[0] == 0:
            sw_preds = sw_preds[1:]
    final_preds[curr:] = sw_preds[0][:(final_preds.shape[0]-curr)]
    
    return final_preds
    

def load_label(label_f, video_len):
    """
    Read label file and generate frame-wise and instance-wise labels accordingly.
    Output:
        labels: Annotated action instances, a list [(start_time, end_time, action_class), ...].
        frame_labels: Frame-wise labels, a [T] numpy array.
    """
    
    labels = []
    frame_labels = np.zeros((video_len), np.int32)
    
    f = open(label_f, 'r')
    annotations = f.readlines()
    
    for ann in annotations:
        ann = ann.strip().split()
        if len(ann) == 0:
            continue
        start_frame = int(ann[0])
        end_frame = int(ann[1])
        action_class = int(ann[2])
        labels.append((start_frame, end_frame, action_class))
        frame_labels[start_frame:(end_frame+1)] = action_class
    f.close()
    
    return labels, frame_labels
    

def visualize_frame(preds, labels, colors):
    """
    Visualize frame-wise predictions along with labels.
    Input:
        preds: Frame-wise classification scores, a [T, num_class] numpy array.
        labels: Frame-wise labels, a [T] numpy array.
        colors: A set of colors to denote the categories.
    """
    
    # Prepare scores and labels in a easy-to-draw format
    pred_scores = np.amax(preds, axis=-1)
    pred_class = np.argmax(preds, axis=-1)
    pred_scores = np.where((pred_class>0), pred_scores, 0)
    label_scores = np.where((labels>0), 1, 0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # Visualize frame-wise labels
    for t in range(labels.shape[0]-1):
        label_x = [t, t+1]
        label_y = [label_scores[t], label_scores[t+1]]
        label_c = colors[int(labels[t+1])]
        ax1.plot(label_x, label_y, color=label_c, linestyle='-', marker='')
        
    # Visualize frame-wise predictions
    for t in range(preds.shape[0]-1):
        pred_x = [t, t+1]
        pred_y = [pred_scores[t], pred_scores[t+1]]
        pred_c = colors[int(pred_class[t+1])]
        ax2.plot(pred_x, pred_y, color=pred_c, linestyle='-', marker='')
    
    # Visualize
    plt.show()
    
    
def accumulate(labels, preds, conf_thresh, num_class=10):
    """
    Accumulate statistics on each category from a video.
    Input:
        labels: A (T, C) numpy array.
        preds: A (T, C) numpy array.
    """
    
    tp_ = np.zeros((num_class, conf_thresh.shape[0]))
    fp_ = np.zeros((num_class, conf_thresh.shape[0]))
    fn_ = np.zeros((num_class, conf_thresh.shape[0]))
    
    p_ = np.sum(labels, axis=0)
    n_ = labels.shape[0] - p_
    # Vary the confidence threshold
    for i, thresh in enumerate(conf_thresh):
        pred_class = np.where((preds > thresh), 1, 0)
        tp_[:, i] = np.sum(np.where(((pred_class+labels)==2), 1, 0), axis=0)
        fp_[:, i] = np.sum(pred_class, axis=0) - tp_[:, i]
        fn_[:, i] = p_ - tp_[:, i]
    
    return tp_, fp_, fn_, p_, n_
    
    
def calculate_cAP(tp, fp, fn, p, n):
    """
    Calculate the calibrated AP on each category.
    Input:
        tp: A (C, N) numpy array.
        fp: A (C, N) numpy array.
        fn: A (C, N) numpy array.
        p: A (C) numpy array.
        n: A (C) numpy array.
    Output:
        cAP: A (C) numpy array.
    """
    
    prec = np.zeros_like(tp)
    recall = np.zeros_like(tp)
    # Negative-Positive ratio
    w = n / p
    # Traverse statistics on each threshold
    for i in range(tp.shape[1]):
        prec[:, i] = tp[:, i] / (tp[:, i] + fp[:, i] / w)
        recall[:, i] = tp[:, i] / (tp[:, i] + fn[:, i])
    # Calculate cAP
    prec = prec[:, ::-1]
    recall = recall[:, ::-1]
    pre_recall = np.zeros_like(w)
    cAP = np.zeros_like(w)
    for i in range(tp.shape[1]):
        cAP += prec[:, i] * (recall[:, i] - pre_recall)
        pre_recall = recall[:, i]
        
    return cAP        
                
        
if __name__ == '__main__':

    # Configs and hyper-params
    SKT_DIR = '/media/home_bak/swap/BigActionData/skeletons_split/val' 
    LABEL_DIR = '/media/home_bak/swap/BigActionData/labels_split/val' 

    SEQ_LEN = 100
    NUM_KP = 10
    DIM_KP = 4
    NUM_CLASS = 10
    SEQ_STEP = 20
    FIRST_STEP = 80
    
    CONF_THRESH = np.linspace(0, 1, 11)[1:-1]

    GPU_MEMORY_FRACTION = 1.0
    MODEL_PATH = 'logs/pose_v3/model.ckpt-49'
    COLORS = np.random.rand(NUM_CLASS+1, 3)
    
    # Load pretrained model
    model = CNN_Model(num_kp=NUM_KP,
                      dim_kp=DIM_KP,
                      seq_len=SEQ_LEN,
                      num_class=NUM_CLASS,
                      gpu_memory_fraction=GPU_MEMORY_FRACTION,
                      training=False)
    model.build_model()
    model.saver.restore(model.sess, MODEL_PATH)
    print ('--Pretrained weights loaded')
    
    # Container for counting cAP on each category
    TP = np.zeros((NUM_CLASS, CONF_THRESH.shape[0]))
    FP = np.zeros((NUM_CLASS, CONF_THRESH.shape[0]))
    FN = np.zeros((NUM_CLASS, CONF_THRESH.shape[0]))
    P = np.zeros((NUM_CLASS))
    N = np.zeros((NUM_CLASS))
    
    for skt_f in os.listdir(SKT_DIR):
        video = skt_f[:-4]
        print (video)
        # Load skeletons
        skt_seq = np.load(os.path.join(SKT_DIR, skt_f), 'r')
        video_len = skt_seq.shape[0]
        # Load labels
        label_f = os.path.join(LABEL_DIR, video+'.txt')
        labels, frame_labels = load_label(label_f, video_len)
        # Transform (T,) labels to (T, C) one-hot labels
        frame_labels = np.eye(NUM_CLASS+1)[frame_labels][:, 1:]
        # Predict classification scores
        #visualize_predict_frame(skt_seq, labels, model, seq_len=SEQ_LEN, seq_step=SEQ_STEP, first_step=FIRST_STEP)
        preds = predict_frame(skt_seq, model, seq_len=SEQ_LEN, seq_step=SEQ_STEP, first_step=FIRST_STEP)
        
        # For each video, accumulate statistics on each category
        tp_, fp_, fn_, p_, n_ = accumulate(frame_labels, preds, CONF_THRESH, NUM_CLASS)
        TP += tp_
        FP += fp_
        FN += fn_
        P += p_
        N += n_
        
    # Calculate the cAP on each category and mean cAP
    cAP = calculate_cAP(TP, FP, FN, P, N)
    print (cAP)
    print (np.mean(cAP))
