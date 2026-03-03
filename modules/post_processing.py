def remove_duplicate_tokens(batch_tensor):
    """
    Remove any token that has already appeared earlier in the same sequence.
    Keeps only the first occurrence of each token.
    """
    batch_list = []
    for seq in batch_tensor.cpu().tolist():
        seen = set()
        new_seq = []
        for token in seq:
            if token not in seen:
                seen.add(token)
                new_seq.append(token)
        batch_list.append(new_seq)
    return batch_list
def convert_to_list(batch_tensor):
    batch_list = []
    for seq in batch_tensor.cpu().tolist():
        new_seq = []
        for token in seq:
            new_seq.append(token)
        batch_list.append(new_seq)
    return batch_list