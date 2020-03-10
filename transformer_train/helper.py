import datetime
import numpy as np
import os
import csv
from sklearn.metrics import precision_recall_fscore_support


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def pad_features(posts_list, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(posts_list), seq_length), dtype=int)

    for i, post in enumerate(posts_list):
        post_len = len(post)

        if post_len <= seq_length:
            zeroes = list(np.zeros(seq_length - post_len))
            new = post + zeroes
        elif post_len > seq_length:
            new = post[0:seq_length - 1]
            new.append(102)

        features[i, :] = np.array(new)

    return features


def save_statistics(experiment_log_dir, filename, stats_dict, current_epoch, continue_from_mode=False):
    """
    Saves the statistics in stats dict into a csv file. Using the keys as the header entries and the values as the
    columns of a particular header entry
    :param experiment_log_dir: the log folder dir filepath
    :param filename: the name of the csv file
    :param stats_dict: the stats dict containing the data to be saved
    :param current_epoch: the number of epochs since commencement of the current training session (i.e. if the experiment continued from 100 and this is epoch 105, then pass relative distance of 5.)
    :param save_full_dict: whether to save the full dict as is overriding any previous entries (might be useful if we want to overwrite a file)
    :return: The filepath to the summary file
    """
    summary_filename = os.path.join(experiment_log_dir, filename)
    mode = 'a' if continue_from_mode else 'w'
    with open(summary_filename, mode) as f:
        writer = csv.writer(f)
        if not continue_from_mode:
            writer.writerow(list(stats_dict.keys()))


        row_to_add = [value[current_epoch] for value in list(stats_dict.values())]
        writer.writerow(row_to_add)

    return summary_filename