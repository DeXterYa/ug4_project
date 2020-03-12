import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import re
from transformers import DistilBertForSequenceClassification, AdamW
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertPreTrainedModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
import time
import random
import os
from helper import format_time, flat_accuracy, pad_features, save_statistics
from arg_extractor import get_args
from sklearn.metrics import precision_recall_fscore_support

args = get_args()

main_data = pd.read_excel('../lstm/stanfordMOOCForumPostsSet.xlsx')

# main_data = pd.read_excel('/home/dexter/ug4_project/data/stanfordMOOCForumPostsSet.xlsx')
sentences = main_data['Text'].tolist()
if args.target == "sentiment":
    sentiment = main_data['Sentiment(1-7)'].tolist()
elif args.target == "confusion":
    sentiment = main_data['Confusion(1-7)'].tolist()
elif args.target == "urgency":
    print("urgency")
    sentiment = main_data['Urgency(1-7)'].tolist()
elif args.target == "combination":
    sentiment = main_data['Sentiment(1-7)'].tolist()
    confusion = main_data['Confusion(1-7)'].tolist()
    urgency = main_data['Urgency(1-7)'].tolist()

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')

tokenizer = DistilBertTokenizer.from_pretrained('../lstm/distiledubert/vocab.txt')
# tokenizer = DistilBertTokenizer.from_pretrained('/home/dexter/Downloads/distiledubert/vocab.txt')

# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased', do_lower_case=True)

input_ids = []
deleted_list = []

for i, sent in enumerate(sentences):
    try:
        if len(sent) == 0:
            print("found zero", i)
        if len(sent) == 1:
            print(sent)
    except:
        print(i)
        deleted_list.append(i)

    if str(sent).endswith("\""):
        sent = str(sent)[:-1].replace('\x07', '').replace('\\', '\\"').replace('\\', '').replace('""', '"')
        sent = re.sub(r"https?\:\/\/[a-zA-Z0-9][a-zA-Z0-9\.\_\?\=\/\%\-\~\&]+", " ", sent)
    else:
        sent = str(sent).replace('\x07', '').replace('\\', '\\"').replace('\\', '').replace('""', '"')
        sent = re.sub(r"https?\:\/\/[a-zA-Z0-9][a-zA-Z0-9\.\_\?\=\/\%\-\~\&]+", " ", sent)

    encoded_sent = tokenizer.encode(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
    )

    input_ids.append(encoded_sent)

# Print sentence 0, now as a list of IDs.
print('Original: ', sentences[0])
print('Token IDs:', input_ids[0])

for idx in deleted_list[::-1]:
    del input_ids[idx]
    del sentiment[idx]
    del confusion[idx]
    del urgency[idx]

input_ids = pad_features(input_ids, 512)

# Create attention masks
attention_masks = []

# For each sentence...
for sent in input_ids:
    # Create the attention mask.
    #   - If a token ID is 0, then it's padding, set the mask to 0.
    #   - If a token ID is > 0, then it's a real token, set the mask to 1.
    att_mask = [int(token_id > 0) for token_id in sent]

    # Store the attention mask for this sentence.
    attention_masks.append(att_mask)

labels_s = [0 if l <= 4 else 1 for l in sentiment]
labels_c = [0 if l <= 4 else 1 for l in confusion]
labels_u = [0 if l <= 4 else 1 for l in urgency]

# Use 90% for training and 10% for validation.
train_inputs, validation_inputs, train_labels_s, validation_labels_s, train_labels_c, validation_labels_c, train_labels_u, validation_labels_u, train_masks, validation_masks = train_test_split(
    input_ids, labels_s, labels_c, labels_u, attention_masks,
    random_state=2020, test_size=0.1)
# Do the same for the masks.


train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)

train_labels_s = torch.tensor(train_labels_s).type(torch.cuda.LongTensor)
validation_labels_s = torch.tensor(validation_labels_s).type(torch.cuda.LongTensor)

train_labels_u = torch.tensor(train_labels_u).type(torch.cuda.LongTensor)
validation_labels_u = torch.tensor(validation_labels_u).type(torch.cuda.LongTensor)

train_labels_c = torch.tensor(train_labels_c).type(torch.cuda.LongTensor)
validation_labels_c = torch.tensor(validation_labels_c).type(torch.cuda.LongTensor)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

batch_size = args.batch_size

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels_s, train_labels_c, train_labels_u)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels_s, validation_labels_c, validation_labels_u)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

torch.cuda.empty_cache()


# config = DistilBertConfig.from_json_file('./distiledubert/config.json')
# bert_model = DistilBertModel.from_pretrained('./distiledubert/pytorch_model.bin', config=config)
class Model(DistilBertPreTrainedModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.num_labels = config.num_labels
        self.distilbert = DistilBertModel(config)
        # self.distilbert = DistilBertModel.from_pretrained(
        #     "/home/dexter/Downloads/distiledubert")
        self.pre_classifier_1 = nn.Linear(config.dim, config.dim)
        self.classifier_1 = nn.Linear(config.dim, config.num_labels)
        self.pre_classifier_2 = nn.Linear(config.dim, config.dim)
        self.classifier_2 = nn.Linear(config.dim, config.num_labels)
        self.pre_classifier_3 = nn.Linear(config.dim, config.dim)
        self.classifier_3 = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()


    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels_s=None,
                labels_c=None, labels_u=None):
        distilbert_output = self.distilbert(
            input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)

        pooled_output_1 = self.pre_classifier_1(pooled_output)  # (bs, dim)
        pooled_output_1 = nn.ReLU()(pooled_output_1)  # (bs, dim)
        pooled_output_1 = self.dropout(pooled_output_1)  # (bs, dim)
        logits_1 = self.classifier_1(pooled_output_1)  # (bs, dim)
        loss_fct = nn.CrossEntropyLoss()


        pooled_output_2 = self.pre_classifier_2(pooled_output)  # (bs, dim)
        pooled_output_2 = nn.ReLU()(pooled_output_2)  # (bs, dim)
        pooled_output_2 = self.dropout(pooled_output_2)  # (bs, dim)
        logits_2 = self.classifier_2(pooled_output_2)  # (bs, dim)


        pooled_output_3 = self.pre_classifier_3(pooled_output)  # (bs, dim)
        pooled_output_3 = nn.ReLU()(pooled_output_3)  # (bs, dim)
        pooled_output_3 = self.dropout(pooled_output_3)  # (bs, dim)
        logits_3 = self.classifier_3(pooled_output_3)  # (bs, dim)

        outputs = (logits_1,logits_2,logits_3)


        if labels_s is not None:
            loss_1 = loss_fct(logits_1.view(-1, 2), labels_s.view(-1))
            loss_2 = loss_fct(logits_2.view(-1, 2), labels_c.view(-1))
            loss_3 = loss_fct(logits_3.view(-1, 2), labels_u.view(-1))

            outputs = (loss_1 + loss_2 + loss_3,) + outputs

        return outputs


# model = DistilBertForSequenceClassification.from_pretrained(
#     "../lstm/distiledubert", # Use the 12-layer BERT model, with an uncased vocab.
#     num_labels = 2, # The number of output labels--2 for binary classification.
#                     # You can increase this for multi-class tasks.
#     output_attentions = False, # Whether the model returns attentions weights.
#     output_hidden_states = False, # Whether the model returns all hidden-states.
# )
model = Model.from_pretrained(
    "../lstm/distiledubert", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()

optimizer = AdamW(model.parameters(),
                  lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                  )

# Number of training epochs (authors recommend between 2 and 4)
epochs = args.num_epochs

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)

device = torch.device("cuda")

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128


# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = []

records = {"curr_epoch": [], "train_loss": [], "val_acc_s": [],"val_acc_c": [],"val_acc_u": [], "prec": [], "recall": [], "fscore": []}
experiment_logs = './logs/' + args.experiment_name + '/'
if not os.path.exists('logs'):
    try:
        os.mkdir('logs')
    except:
        print('logs exist')
if not os.path.exists(experiment_logs):
    try:
        os.mkdir(experiment_logs)
    except:
        print('experiment directory exists')

fscore_check = 0
# For each epoch...
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0

    # Put the model into training mode. Don't be mislead--the call to
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels_s = batch[2].to(device)
        b_labels_c = batch[3].to(device)
        b_labels_u = batch[4].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(b_input_ids,
                        attention_mask=b_input_mask,
                        labels_s=b_labels_s, labels_c=b_labels_c, labels_u=b_labels_u)

        # The call to `model` always returns a tuple, so we need to pull the
        # loss value out of the tuple.
        loss = outputs[0]
        print(loss.item())

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy_s, eval_accuracy_c, eval_accuracy_u = 0, 0, 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    logits_list = []
    labels_list = []

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels_s,b_labels_c, b_labels_u = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids,
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits_s = outputs[0]
        logits_c = outputs[1]
        logits_u = outputs[2]
        # Move logits and labels to CPU
        logits_s = logits_s.detach().cpu().numpy()
        label_ids_s = b_labels_s.to('cpu').numpy()

        logits_c = logits_c.detach().cpu().numpy()
        label_ids_c = b_labels_c.to('cpu').numpy()

        logits_u = logits_u.detach().cpu().numpy()
        label_ids_u = b_labels_u.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy_s = flat_accuracy(logits_s, label_ids_s)
        tmp_eval_accuracy_c = flat_accuracy(logits_c, label_ids_c)
        tmp_eval_accuracy_u = flat_accuracy(logits_u, label_ids_u)

        # Accumulate the total accuracy.
        eval_accuracy_s += tmp_eval_accuracy_s
        eval_accuracy_c += tmp_eval_accuracy_c
        eval_accuracy_u += tmp_eval_accuracy_u

        # Track the number of batches
        nb_eval_steps += 1

        logits_list += list(np.argmax(logits_s, axis=1).flatten()) + list(np.argmax(logits_c, axis=1).flatten()) + list(np.argmax(logits_u, axis=1).flatten())
        labels_list += list(label_ids_s)+list(label_ids_c)+list(label_ids_u)

        # Report the final accuracy for this validation run.
    print("  Accuracy_s: {0:.2f}".format(eval_accuracy_s / nb_eval_steps))
    print("  Accuracy_c: {0:.2f}".format(eval_accuracy_c / nb_eval_steps))
    print("  Accuracy_u: {0:.2f}".format(eval_accuracy_u / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
    prec, recall, fscore, _ = precision_recall_fscore_support(labels_list, logits_list, average='binary')

    records["curr_epoch"].append(epoch_i)
    records["train_loss"].append(avg_train_loss)
    records["val_acc_s"].append(eval_accuracy_s / nb_eval_steps)
    records["val_acc_c"].append(eval_accuracy_c / nb_eval_steps)
    records["val_acc_u"].append(eval_accuracy_u / nb_eval_steps)
    records["prec"].append(prec)
    records["recall"].append(recall)
    records["fscore"].append(fscore)

    if epoch_i == 0:
        save_statistics(experiment_log_dir=experiment_logs, filename="DistilBert_" + args.target + ".csv",
                        stats_dict=records, current_epoch=epoch_i, continue_from_mode=False)
    else:
        save_statistics(experiment_log_dir=experiment_logs, filename="DistilBert_" + args.target + ".csv",
                        stats_dict=records, current_epoch=epoch_i, continue_from_mode=True)

    output_dir = './model_save/'

    if fscore >= fscore_check:
        fscore_check = fscore

        # Create output directory if needed
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except:
                print('dir 1 exists')

        output_dir = './model_save/' + 'DistilBert_' + args.target + '/'

        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except:
                print('dir 2 exists')

        print("Saving model to %s" % output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

print("")
print("Training complete!")

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()


# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))
