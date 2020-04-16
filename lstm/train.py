import csv
import spacy
from spacy.lang.en import English
from arg_extractor import get_args
import pickle
import os
import bcolz
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from model import Model
from numpy.random import default_rng
from process import get_sequences, load_obj, save_statistics
from data_provider import data_provider
torch.cuda.empty_cache()
args = get_args()
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)



if torch.cuda.is_available():
    torch.device('cuda')

if args.update_emb is False:
    print("Do not update embedding vectors")

nlp = spacy.load('en')
if args.stopwords is False:
    print("remove stop words from posts")
    nlp.vocab["br"].is_stop = True
    nlp.vocab["href"].is_stop = True
    nlp.vocab["\\"].is_stop = True
    nlp.vocab["li"].is_stop = True
    nlp.vocab["div"].is_stop = True
    nlp.vocab["br"].is_stop = True
    nlp.vocab["span"].is_stop = True
tokenizer = English().Defaults.create_tokenizer(nlp)

num_epochs = args.num_epochs
LEARNING_RATE = args.lr
EMBEDDING_DIM = 300


# Load word vectors
vectors = bcolz.open(f'./glove.6B/6B.300d.dat')[:]
word2idx = pickle.load(open(f'./glove.6B/6B.300d_idx.pkl', 'rb'))
vec = torch.FloatTensor(vectors)
vocab = word2idx
# vec = torch.load( '../fasttext_vec.pt')
# vocab = load_obj('../obj/fasttext')
vec = vec.cuda()
max_idx = len(vocab)



# Handle OOV words: out-of-vocabulary
for word in ['<unk>', '<pad>']:
    k = np.random.rand(1, EMBEDDING_DIM) # Generate a random 1*300 vector
    k = 7*k/np.linalg.norm(k) # Normalize the vector
    vocab[word]= max_idx
    k_tensor = torch.from_numpy(k).cuda()
    k_tensor = k_tensor.type(torch.cuda.FloatTensor)
    vec = torch.cat((k_tensor,vec), 0)
    max_idx += 1


lstm_args = {}
lstm_args['embed_num'] = max_idx
lstm_args['vec'] = vec
lstm_args['class_num'] = 2
lstm_args['cuda'] = torch.cuda.is_available()
lstm_args['hidden'] = args.num_hidden
lstm_args['embed_dim'] = EMBEDDING_DIM
lstm_args['dropout'] = args.dropout


# Intialise model
lstm = Model(lstm_args)
lstm = lstm.cuda()

optimizer = torch.optim.Adam(lstm.parameters(), lr=LEARNING_RATE)

threads, labels, features = data_provider(args.dataset_name)


# Extract features which are useful
X = features.to_numpy()[:,2:]
t = [ 0,1,5,7,8,14,15]
X = X[:, t]
a = X[:, 2]
X[:, 2] = np.asarray([float(i - min(a))/(max(a)-min(a)) for i in a])

# Split training data and testing data
rng = default_rng(seed = 0)
num_tv = int(0.9 * len(threads))
idx = rng.choice(len(threads), size=num_tv, replace=False)

# Data for training and validation
X_tv = X[idx,:]
threads_tv = list( threads[i] for i in idx )
labels_tv = list( labels[i] for i in idx )

# Data for training
num_train = num_tv -(len(threads) - num_tv)
train_idx = [i for i in range(0,num_train)]
X_train = X_tv[0: num_train, :]
X_train = torch.from_numpy(X_train.astype('float64')).float().cuda()
threads_train = list( threads_tv[i] for i in train_idx )
labels_train = list( labels_tv[i] for i in train_idx )

y_train = labels_train

# Data for validation
X_valid = X_tv[num_train:, :]
X_valid = torch.from_numpy(X_valid.astype('float64')).float().cuda()
valid_idx = [i for i in range(num_train, num_tv)]
threads_valid = list( threads_tv[i] for i in valid_idx )
labels_valid = list( labels_tv[i] for i in valid_idx )


# Calculate class ratio
class_ratio = np.bincount(y_train)
print(class_ratio)
class_weight_dict = { 0: 1.0,
                      1: class_ratio[0]*1.0 /class_ratio[1]
                    }
class_weights_tensor = torch.FloatTensor([class_weight_dict[1]]).cuda()
print(class_weights_tensor)

# Evaluation
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)
def evaluation():
    y_preds = []
    y_outputs = []
    y_true = []
    loss_items = []
    for thread_idx in range(len(threads_valid)):

        original_thread_length = len(threads_valid[thread_idx])
        if original_thread_length == 0:
            continue
        word_idxs = get_sequences(threads_valid[thread_idx], tokenizer, vocab)

        word_idxs_tensor = torch.LongTensor(word_idxs)
        inp = Variable(word_idxs_tensor, requires_grad=False).cuda()

        output = lstm(inp, X_valid[thread_idx])
        targets = [labels_valid[thread_idx]]
        targets_tensor = torch.FloatTensor(targets).view(1, -1)
        target = Variable(targets_tensor, requires_grad=False).cuda()
        loss = criterion(output, target)
        loss_items.append(loss.item())
        y_outputs.append(output)

        #     _,prediction = op.max(dim=1)
        #     prediction  = prediction.item()
        if float(output) < 0:
            prediction = 0
        else:
            prediction = 1

        y_preds.append(prediction)
        y_true.append(labels_valid[thread_idx])

    prec, recall, fscore, _ = precision_recall_fscore_support(y_true, y_preds, average='binary')
    loss = (sum(loss_items)/len(loss_items))
    print('Loss, Precision, Recall, F-score', loss, prec, recall, fscore)
    return loss, prec, recall, fscore

records = { "curr_epoch": [], "train_loss": [], "val_loss": [], "prec": [], "recall": [],"fscore": []}
experiment_logs = './logs/'+args.experiment_name+'/'
if not os.path.exists('logs'):
    os.mkdir('logs')
if not os.path.exists(experiment_logs):
    os.mkdir(experiment_logs)


# Train

best_performance = 0.0
best_prec = 0.0
best_recall = 0.0
best_fscore = 0.0
best_epoch = 0
best_loss = 0
for epoch in range(1, num_epochs + 1):
    loss_items = []
    for thread_idx in range(len(X_train)):
        lstm.train()
        targets = [y_train[thread_idx]]
        targets_tensor = torch.FloatTensor(targets).view(1, -1)
        target = Variable(targets_tensor, requires_grad=False).cuda()

        original_thread_length = len(threads_train[thread_idx])
        if original_thread_length == 0:
            continue
        word_idxs = get_sequences(threads_train[thread_idx], tokenizer, vocab)

        if word_idxs.size == 0:
            continue

        word_idxs_tensor = torch.LongTensor(word_idxs)
        inp = Variable(word_idxs_tensor, requires_grad=False).cuda()

        # Forward pass
        output = lstm(inp, X_train[thread_idx])

        loss = criterion(output, target)

        #         loss = F.cross_entropy(logit, target, size_average=False)

        # Zero the gradients before running the backward pass.
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        loss_items.append(loss.item())

    print("epoch:", epoch, "  loss:", sum(loss_items)/len(loss_items))
    records["curr_epoch"].append(epoch)
    records["train_loss"].append(sum(loss_items) / len(loss_items))
    lstm.eval()
    loss_val, prec, recall, fscore = evaluation()
    records["val_loss"].append(loss_val)
    records["prec"].append(prec)
    records["recall"].append(recall)
    records["fscore"].append(fscore)

    if epoch == 1:
        print("records",records)
        print("epoch",epoch)
        save_statistics(experiment_log_dir=experiment_logs, filename=args.dataset_name + "_" + str(args.seed) + ".csv",
                        stats_dict=records, current_epoch=epoch, continue_from_mode=False)
    else:
        save_statistics(experiment_log_dir=experiment_logs, filename=args.dataset_name + "_" + str(args.seed) + ".csv",
                        stats_dict=records, current_epoch=epoch, continue_from_mode=True)
    if fscore > best_performance:
        best_performance = fscore
        best_epoch, best_loss, best_prec, best_recall, best_fscore = epoch, loss_val, prec, recall, fscore



print(best_prec, best_recall, best_fscore)

file_name = args.experiment_name
flag = 0
try:
    f = open('./results/'+ file_name + '.csv')
    # Do something with the file
except IOError:
    flag = 1
with open('./results/'+ file_name + '.csv', mode='a') as csv_file:
    fieldnames = ['dataset_name', 'seed', 'num_train', 'num_valid', 'best_epoch', 'loss', 'best_prec', 'best_recall',
                  'best_fscore']
    if flag == 1:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'dataset_name': args.dataset_name, 'seed': args.seed, 'num_train': num_train,
                         'num_valid': (len(threads) - num_tv),
                         'best_epoch': best_epoch, 'loss': best_loss, 'best_prec': best_prec,
                         'best_recall': best_recall, 'best_fscore': best_fscore})
    else:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({'dataset_name': args.dataset_name, 'seed': args.seed, 'num_train': num_train, 'num_valid':(len(threads) - num_tv),
                     'best_epoch': best_epoch, 'loss': best_loss, 'best_prec': best_prec, 'best_recall': best_recall, 'best_fscore': best_fscore})




