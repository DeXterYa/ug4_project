import csv
from arg_extractor import get_args
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
from model import Model
from numpy.random import default_rng
from gensim.models import KeyedVectors
from process import get_sequences, load_obj
from data_provider import data_provider
import numpy as np
import torch
from torch import nn
from transformers import DistilBertModel, DistilBertTokenizer,DistilBertConfig
import logging
torch.cuda.empty_cache()
args = get_args()
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
import datetime

currentDT = datetime.datetime.now()
print(str(currentDT))
logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(logging.ERROR)


if torch.cuda.is_available():
    torch.device('cuda')

if args.update_emb is False:
    print("Do not update embedding vectors")

config = DistilBertConfig.from_json_file('/home/dexter/Downloads/distiledubert/config.json')
bert_model = DistilBertModel.from_pretrained('/home/dexter/Downloads/distiledubert/pytorch_model.bin', config=config)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')




num_epochs = args.num_epochs
LEARNING_RATE = args.lr
EMBEDDING_DIM = 300




lstm_args = {}
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
for i, thread in enumerate(threads):
    threads[i] = get_sequences(thread, tokenizer)

attention_masks = []

# For each sentence...
for thread in threads:
    att_mask_posts = []
    for post in thread:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in post]
        att_mask_posts.append(att_mask)

    # Store the attention mask for this sentence.
    attention_masks.append(att_mask_posts)


# Extract features which are useful
X = features.as_matrix(columns=features.columns[2:])
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
attention_masks_tv = list( attention_masks[i] for i in idx)
labels_tv = list( labels[i] for i in idx )

# Data for training
num_train = num_tv -(len(threads) - num_tv)
train_idx = [i for i in range(0,num_train)]
X_train = X_tv[0: num_train, :]
X_train = torch.from_numpy(X_train.astype('float64')).float().cuda()
threads_train = list( threads_tv[i] for i in train_idx )
labels_train = list( labels_tv[i] for i in train_idx )
attention_masks_train = list(attention_masks_tv[i] for i in train_idx)

y_train = labels_train

# Data for validation
X_valid = X_tv[num_train:, :]
X_valid = torch.from_numpy(X_valid.astype('float64')).float().cuda()
valid_idx = [i for i in range(num_train, num_tv)]
threads_valid = list( threads_tv[i] for i in valid_idx )
labels_valid = list( labels_tv[i] for i in valid_idx )
attention_masks_valid = list( attention_masks_tv[i] for i in valid_idx )


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
        word_idxs = threads_valid[thread_idx]

        word_idxs_tensor = torch.LongTensor(word_idxs)

        inp = Variable(word_idxs_tensor, requires_grad=False).cuda()

        try:
            bert_output = bert_model(inp)[0][:, 0, :]

            # Forward pass
            output = lstm(bert_output, X_valid[thread_idx])
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
            del bert_output, output, loss, inp, target
        except:
            print('cuda problem in evaluation')
            del inp
        torch.cuda.empty_cache()

    prec, recall, fscore, _ = precision_recall_fscore_support(y_true, y_preds, average='binary')
    loss = (sum(loss_items)/len(loss_items))
    print('Loss, Precision, Recall, F-score', loss, prec, recall, fscore)
    return loss, prec, recall, fscore


for param in bert_model.parameters():
    param.requires_grad = False

bert_model.cuda()

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
        word_idxs = threads_train[thread_idx]

        if word_idxs.size == 0:
            continue

        word_idxs_tensor = torch.LongTensor(word_idxs)
        inp = Variable(word_idxs_tensor, requires_grad=False).cuda()

        try:
            bert_output = bert_model(inp)[0][:,0,:]

            # Forward pass
            output = lstm(bert_output, X_train[thread_idx])

            loss = criterion(output, target)

            #         loss = F.cross_entropy(logit, target, size_average=False)

            # Zero the gradients before running the backward pass.
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            loss_items.append(loss.item())
            del bert_output, output, loss, inp, target
        except:
            print("cuda problem")
            del inp, target

        torch.cuda.empty_cache()

    print("epoch:", epoch, "  loss:", sum(loss_items)/len(loss_items))
    lstm.eval()
    loss_val, prec, recall, fscore = evaluation()
    if fscore > best_performance:
        best_performance = fscore
        best_epoch, best_loss, best_prec, best_recall, best_fscore = epoch, loss_val, prec, recall, fscore



print(best_prec, best_recall, best_fscore)

file_name = args.experiment_name
flag = 0
try:
    f = open('/home/dexter/ug4_project/lstm/results/'+ file_name + '.csv')
    # Do something with the file
except IOError:
    flag = 1
with open('/home/dexter/ug4_project/lstm/results/'+ file_name + '.csv', mode='a') as csv_file:
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


currentDT = datetime.datetime.now()
print(str(currentDT))

