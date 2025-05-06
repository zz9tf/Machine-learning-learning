from torch.nn.utils.rnn import pad_sequence

# 假设你已经把文本转成了 token index 的 list
tokenized_texts = [[1, 4, 5], [7, 8], [3, 6, 9, 10]]  # 举例

# Padding（注意：要变成 tensor，且需要 pack_padded_sequence 时提供长度）
lengths = torch.tensor([len(seq) for seq in tokenized_texts])
padded_seqs = pad_sequence([torch.tensor(seq) for seq in tokenized_texts], batch_first=True, padding_value=0)
