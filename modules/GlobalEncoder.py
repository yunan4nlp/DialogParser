import torch.nn as nn
import torch

class WordEncoder(nn.Module):
    def __init__(self, vocab, config):
        super(WordEncoder, self).__init__()
        self.config = config

        self.word_embeddings = nn.Embedding(vocab.word_size, config.word_dims, vocab.PAD)
        self.extword_embeddings = nn.Embedding(vocab.extword_size, config.word_dims, vocab.PAD)

        self.word_drop = nn.Dropout(config.dropout_emb)

        self.word_GRU = nn.GRU(input_size=config.word_dims,
                               hidden_size=config.gru_hiddens // 2,
                               num_layers=config.gru_layers,
                               bidirectional=True, batch_first=True)

        self.hidden_drop = nn.Dropout(config.dropout_gru_hidden)

    def initial_pretrain(self, vec):
        self.extword_embeddings.weight.data.copy_(torch.from_numpy(vec))
        self.extword_embeddings.weight.requires_grad = False


    def forward(self, word_indexs, extword_indexs, word_lengths):
        batch_size, max_edu_num, max_edu_len = word_indexs.size()

        word_indexs = word_indexs.view(-1, max_edu_len)
        extword_indexs = extword_indexs.view(-1, max_edu_len)

        word_embs = self.word_embeddings(word_indexs)

        extword_embs = self.extword_embeddings(extword_indexs)

        word_represents = self.word_drop(word_embs + extword_embs)

        gru_input = nn.utils.rnn.pack_padded_sequence(word_represents, word_lengths,
                                                      batch_first=True, enforce_sorted=False)

        _, hidden = self.word_GRU(gru_input)

        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)

        hidden = hidden.view(batch_size, max_edu_num, -1)

        hidden = self.hidden_drop(hidden)

        return hidden

class EDUEncoder(nn.Module):
    def __init__(self, vocab, config):
        super(EDUEncoder, self).__init__()
        self.config = config

        self.edu_GRU = nn.GRU(input_size=config.gru_hiddens,
                              hidden_size=config.gru_hiddens,
                              num_layers=config.gru_layers,
                              bidirectional=False, batch_first=True)
        #self.hidden_drop = nn.Dropout(config.dropout_gru_hidden)

    def forward(self, edu_hidden, edu_lengths):
        gru_input = nn.utils.rnn.pack_padded_sequence(edu_hidden, edu_lengths, batch_first=True, enforce_sorted=False)

        outputs, _ = self.edu_GRU(gru_input)

        outputs = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        #hidden = self.hidden_drop(outputs[0])
        hidden = outputs[0]

        return hidden

class GlobalEncoder(nn.Module):
    def __init__(self, vocab, config):
        super(GlobalEncoder, self).__init__()

        self.word_encoder = WordEncoder(vocab, config)
        self.EDU_encoder = EDUEncoder(vocab, config)

    def forward(self, word_indexs, extword_indexs, word_lengths, edu_lengths):

        edu_represents = self.word_encoder(word_indexs, extword_indexs, word_lengths)
        edu_outputs = self.EDU_encoder(edu_represents, edu_lengths)

        return edu_represents, edu_outputs
