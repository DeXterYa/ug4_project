import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from arg_extractor import get_args
global_args = get_args()
class Attn(nn.Module):

    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(2*self.hidden_size, self.hidden_size)
        #create new parameter for attn weights
        self.v = nn.Parameter(torch.rand(hidden_size))
        #initialise attn weight parameter
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden,encoder_outputs):
        max_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        #make batch first in the lstm outputs
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        #unormalised attention scores
        attn_energies = self.score(H,encoder_outputs) # compute attention score for each context
        return F.softmax(attn_energies).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
#         cat = torch.cat([hidden, encoder_outputs], 2)
        #print(cat.size())
        #exit()
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        V = args['embed_num']
        D = args['embed_dim']
        C = args['class_num']
        Ci = 1
        self.hidden = args['hidden']

        self.embed = nn.Embedding(V, D)
        self.embed.weight = nn.Parameter(args['vec'])
        print('Allow updating embedding:',global_args.update_emb)
        self.embed.weight.requires_grad = global_args.update_emb  # Allow to update the embedding vectors
        # hidden size is the size of the output tensor
        self.lstm1 = nn.LSTM(input_size=D, hidden_size=self.hidden, num_layers=2, dropout=0.2, bidirectional=False)

        # Compute attention, the output is tensors representing contexts
        self.attn = Attn(self.hidden)
        # input is each context
        self.lstm2 = nn.LSTM(input_size=self.hidden, hidden_size=self.hidden, num_layers=2, dropout=0.2, bidirectional=False)
        self.dropout = nn.Dropout(args['dropout'])
        # fully connected layer
        self.fc1 = nn.Linear(self.hidden, 1)

    #         with torch.no_grad():
    #             self.linear.weight.copy_(your_new_weights)

    def forward(self, x, extra):
        x = self.embed(x)  # (N,W,D) N: number of posts W: number of words D: embedding dimension
        x = Variable(x)  # Not necessary
        x = x.permute(1, 0, 2)  # (W,N,D)
        x, _ = self.lstm1(x)  # (W,N,H) the output of last layer in each t
        post_vectors = x[-1]  # (N,H) the last output of each post
        post_vectors = post_vectors.unsqueeze(1)  # (N,1,H)

        context_representations, _ = self.lstm2(post_vectors)  # (N,1,H) different context representation

        # attn_weights = self.attn(post_vectors[-1],context_representations[0:-1])
        # attended_context_representation = attn_weights.bmm(context_representations[0:-1].transpose(0,1))

        attn_weights = self.attn(post_vectors[-1],
                                 context_representations)  # Use last context as the query to compute weights
        # Use weights to determine the final context representation
        attended_context_representation = attn_weights.bmm(
            context_representations.transpose(0, 1))  # Output a single vector

        # print(attended_context_representation[-1].size())
        # print(post_vectors[-1].size())
        # exit()
        # extra = extra.view(1, -1)

        output = self.dropout(attended_context_representation[-1]).view(1, -1)

        # logits = torch.cat((output, extra), 1)
        logits = output
        logit = self.fc1(logits)  # concatenation and dropout

        return logit