from torch.utils.data import Dataset


class DUIEDataset(Dataset):

    def __init__(self, token_ids, attention_mask, seq_len, token_start_index,
                 token_end_index, labels, **kwargs):
        super(DUIEDataset, self).__init__(**kwargs)
        self.token_ids = token_ids
        self.attention_mask = attention_mask
        self.seq_len = seq_len
        self.token_start_index = token_start_index
        self.token_end_index = token_end_index
        self.labels = labels

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, item):
        return self.token_ids[item], self.attention_mask[item], self.seq_len[item], self.token_start_index[item], \
               self.token_end_index[item], self.labels[item]
