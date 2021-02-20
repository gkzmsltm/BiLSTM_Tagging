import torch.nn as nn
from torchcrf import CRF

class BiLSTMCRF(nn.Module):
    def __init__(self, num_words, embedding_dim, size_hidden, num_poss, num_layers, padding_idx):
        super(BiLSTMCRF, self).__init__()
        self.padding_idx = padding_idx
        self.word_embedding = nn.Embedding(num_words, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, size_hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.25)
        self.linear = nn.Linear(size_hidden*2, num_poss)
        self.crf = CRF(num_poss, batch_first=True)
        self.crf.reset_parameters()
        
    def forward(self, input_seq, target_seq=None, lossmode=True):
        # print(input_seq.size())
        # batch-seq(1 feature)

        output_seq = self.word_embedding(input_seq)
        # print(output_seq.size())
        # batch-seq-embedding_dim

        output_seq, (h_n, c_n) = self.lstm(output_seq)
        # print(output_seq.size())
        # batch-seq-size_hidden*2(BiDirection)
        # print(h_n.size())
        # 2(BiDirection)-batch-size_hidden

        output = self.linear(self.dropout(output_seq))
        # print(output.size())
        # batch-seq-num_poss
        
        # return output #, output_seq, (h_n, c_n)
        mask = input_seq.ne(self.padding_idx)
        if self.training or lossmode:
            loss = -self.crf(output, target_seq, reduction='mean', mask=mask)
            return loss
        else:
            # print('e')
            y = self.crf.decode(output)
            return y