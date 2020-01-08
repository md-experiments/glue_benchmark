
import torch.nn as nn
import torch

class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        
        self.bert = bert
        
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [batch size, sent len]
                
        with torch.no_grad():
            embedded = bert(text)[0]
                
        #embedded = [batch size, sent len, emb dim]
        
        _, hidden = self.rnn(embedded)
        
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden)
        
        #output = [batch size, out dim]
        
        return output


class BERTGRUSimilarity(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        
        self.bert = bert
        
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 1)

    def forward(self, text1, text2):
        
        #text = [batch size, sent len]
                
        with torch.no_grad():
            embedded1 = bert(text1)[0]
            embedded2 = bert(text2)[0]   
        #embedded = [batch size, sent len, emb dim]
        
        _, hidden1 = self.rnn(embedded1)
        _, hidden2 = self.rnn(embedded2)
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden1 = self.dropout(torch.cat((hidden1[-2,:,:], hidden1[-1,:,:]), dim = 1))
            hidden2 = self.dropout(torch.cat((hidden2[-2,:,:], hidden2[-1,:,:]), dim = 1))
        else:
            hidden1 = self.dropout(hidden1[-1,:,:])
            hidden2 = self.dropout(hidden2[-1,:,:])
        #hidden = [batch size, hid dim]
        
        output1 = self.out(hidden1)
        output2 = self.out(hidden2)
        #output = [batch size, out dim]

        logits = -self.__batch_dist__(output1, output2)
        #_, pred = torch.max(logits.view(-1, N), 1)        
        return logits
