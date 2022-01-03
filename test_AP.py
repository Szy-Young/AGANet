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
    
    
def predict_trigger(pred_class, trigger_thresh, accum_limit, cool_down=50, num_class=10):
    """
    Trigger action instances based on frame-wise classification results.
    Input:
        pred_class: Frame-wise classification results, a [T] numpy array.
    Output:
        final_preds: Triggered action instances, a list [(trigger_time, action_class), ...].
    """
    
    final_preds = []
    # Construct accumulator and trigger to record the state
    accumulator = [0 for c in range(num_class)]
    trigger = [0 for c in range(num_class)]
    # Construct cool down counter to avoid repetitive trigger
    cool_count = [0 for c in range(num_class)]
    
    for t in range(pred_class.shape[0]):
        pred_c = pred_class[t]
        # Given predicted class of a frame, update accumulator and trigger state
        for c in range(num_class):
            if cool_count[c] > 0:
                # Decrease the cool down counter
                cool_count[c] -= 1
            if pred_c == (c+1):
                # Accumulate +1 for action duration
                if accumulator[c] < accum_limit[c]:
                    accumulator[c] += 1
                # Trigger an action instance
                if accumulator[c] >= trigger_thresh[c] and trigger[c] == 0 and cool_count[c] == 0:
                    trigger[c] = 1
                    cool_count[c] = cool_down
                    final_preds.append((t, pred_c))
            else:
                # Accumulate -1 for action gap
                if accumulator[c] > 0:
                    accumulator[c] -= 1
                # Reset trigger state to wait for next trigger
                if accumulator[c] < trigger_thresh[c] and trigger[c] == 1:
                    trigger[c] = 0
    
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
    
    
def frame_acc(preds, labels):
    """
    Calculate the frame-wise classification accuracy given labels and classification scores.
    Input:
        preds: Frame-wise classification results, a [T] numpy array.
        labels: Frame-wise labels, a [T] numpy array.
    Output:
        acc: The frame-wise classification accuracy on the sequence.
    """
    
    # Calculate the accuracy
    acc = np.equal(preds, labels).astype(np.float32)
    acc = np.mean(acc)
        
    return acc
    
    
def trigger_acc(preds, labels, num_class=10, allow_delay=0.2):
    """
    Calculate the instance-wise classification accuracy and recall given labels and triggered instances.
    Input:
        preds: Triggered action instances, a list [(trigger_time, action_class), ...].
        labels: Annotated action instances, a list [(start_time, end_time, action_class), ...].
    Output:
        TP: The number of true positive instances, an integer.
    """
    
    # Order the labels by class
    class_label = [[] for c in range(num_class)]
    for label in labels:
        cls = int(label[2])
        class_label[cls-1].append(label)
        
    # Match triggered instances with labels
    TP = 0
    for pred in preds:
        trigger_time = int(pred[0])
        cls = int(pred[1])
        for label in class_label[cls-1]:
            start_time = int(label[0])
            end_time = int(label[1])
            end_time += int(allow_delay * (end_time - start_time + 1))
            if trigger_time >= start_time and trigger_time <= end_time:
                TP += 1
                class_label[cls-1].remove(label)
                break
    
    return TP
    

def trigger_error_stats(preds, labels, num_class=10, allow_delay=0.2):
    """
    For each class separately calculate the instance-wise classification accuracy and recall given labels and 
    triggered instances.
    Input:
        preds: Triggered action instances, a list [(trigger_time, action_class), ...].
        labels: Annotated action instances, a list [(start_time, end_time, action_class), ...].
    Output:
        TP: The number of true positive instances for each class, a list of integers.
        FP: The number of false positive instances for each class, a list of integers.
        FN: The number of false negative instances for each class, a list of integers.
    """
    
    # Order the labels by class
    class_label = [[] for c in range(num_class)]
    for label in labels:
        cls = int(label[2])
        class_label[cls-1].append(label)
    
    # Order the preds by class
    class_pred = [[] for c in range(num_class)]
    for pred in preds:
        cls = int(pred[1])
        class_pred[cls-1].append(pred)
        
    TP = [0 for c in range(num_class)]
    FP = [0 for c in range(num_class)]
    FN = [0 for c in range(num_class)]
    for c in range(num_class):
        FP[c] = len(class_pred[c])
        FN[c] = len(class_label[c])
        
    # Match triggered instances with labels
    for pred in preds:
        trigger_time = int(pred[0])
        cls = int(pred[1])
        for label in class_label[cls-1]:
            start_time = int(label[0])
            end_time = int(label[1])
            end_time += int(allow_delay * (end_time - start_time + 1))
            if trigger_time >= start_time and trigger_time <= end_time:
                TP[cls-1] += 1
                FP[cls-1] -= 1
                FN[cls-1] -= 1
                class_label[cls-1].remove(label)
                break
    
    return TP, FP, FN
    

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
    
    
def visualize_trigger(preds, labels, colors):
    """
    Visualize frame-wise predictions along with labels.
    Input:
        preds: Instance-wise action triggers, a list [(trigger_time, action_class), ...].
        labels: Frame-wise labels, a [T] numpy array.
        colors: A set of colors to denote the categories.
    """
    
    # Prepare labels in a easy-to-draw format
    label_scores = np.where((labels>0), 1, 0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # Visualize frame-wise labels
    for t in range(labels.shape[0]-1):
        label_x = [t, t+1]
        label_y = [label_scores[t], label_scores[t+1]]
        label_c = colors[int(labels[t+1])]
        ax1.plot(label_x, label_y, color=label_c, linestyle='-', marker='')
        
    # Visualize instance-wise action triggers
    for t in range(labels.shape[0]-1):
        trigger = 0
        for pred in preds:
            trigger_time = pred[0]
            if trigger_time == (t+1):
                pred_x = [t, t+1]
                pred_y = [0, 1]
                pred_c = colors[pred[1]]
                trigger = 1
                break
        if trigger == 0:
            pred_x = [t, t+1]
            pred_y = [0, 0]
            pred_c = colors[0]
        ax2.plot(pred_x, pred_y, color=pred_c, linestyle='-', marker='')
    
    # Visualize
    plt.show()
             
        
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
    
    MIN_LENS = [15, 12, 15, 16, 16, 16, 16, 25, 25, 13]
    TRIGGER_THRESH = [int(0.5*min_len) for min_len in MIN_LENS]
    ACCUM_LIMIT = MIN_LENS
    COOL_DOWN = 50  
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
    
    # Container for counting the accuracy on the whole dataset
    total_triggers = [0 for i in range(CONF_THRESH.shape[0])]
    total_labels = 0
    p_triggers = [0 for i in range(CONF_THRESH.shape[0])]
    
    for skt_f in os.listdir(SKT_DIR):
        video = skt_f[:-4]
        print (video)
        # Load skeletons
        skt_seq = np.load(os.path.join(SKT_DIR, skt_f), 'r')
        video_len = skt_seq.shape[0]
        # Load labels
        label_f = os.path.join(LABEL_DIR, video+'.txt')
        labels, frame_labels = load_label(label_f, video_len)
        # Predict classification scores
        #visualize_predict_frame(skt_seq, labels, model, seq_len=SEQ_LEN, seq_step=SEQ_STEP, first_step=FIRST_STEP)
        preds = predict_frame(skt_seq, model, seq_len=SEQ_LEN, seq_step=SEQ_STEP, first_step=FIRST_STEP)
        
        # Vary the confidence threshold to calculate AP
        for i, thresh in enumerate(CONF_THRESH):
            pred_class = np.argmax(preds, axis=-1) + 1
            class_score = np.max(preds, axis=-1)
            pred_class = np.where((class_score>thresh), pred_class, 0)
            # Trigger action instances
            triggers = predict_trigger(pred_class, trigger_thresh=TRIGGER_THRESH, accum_limit=ACCUM_LIMIT, cool_down=COOL_DOWN, num_class=NUM_CLASS)
            TP = trigger_acc(triggers, labels)
            # Accumulate statistics on the whole dataset
            total_triggers[i] += len(triggers)
            p_triggers[i] += TP
        total_labels += len(labels)
    
    # Collect statistics to calculate Precision and Recall
    precisions = [0 for i in range(CONF_THRESH.shape[0])]
    recalls = [0 for i in range(CONF_THRESH.shape[0])]
    for i in range(CONF_THRESH.shape[0]):
        tp = p_triggers[i]
        if total_triggers[i] == 0:
            precisions[i] = 0
        else:
            precisions[i] = tp / total_triggers[i]
        recalls[i] = tp / total_labels
    
    # Interpolate
    interpolate_precisions = []
    last_max = 0
    for precision in precisions:
        interpolate_precisions.append(max(last_max, precision))
        last_max = max(interpolate_precisions)
    
    # Calculate AP
    precisions = precisions[::-1]
    recalls = recalls[::-1]
    pre_recall = 0
    ap = 0
    for precision, recall in zip(precisions, recalls):
        ap += precision * (recall - pre_recall)
        pre_recall = recall
    print ('AP on the whole dataset: %f' %(ap))
    print (precisions)
    print (recalls)
    # Draw P-R curve
    plt.xlim(0.75, 1)
    plt.ylim(0.75, 1)
    plt.plot(recalls, precisions, color='r', linestyle='-', marker='')
    plt.show()
