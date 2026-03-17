import re
def remove_duplicate_tokens(batch_tensor):
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
def remove_duplicate_in_sentence(text: str) -> str:   
    if not text:
        return text
    sentences = text.split(".")
    cleaned_sentences = []
    for sent in sentences:
        if not sent.strip():
            cleaned_sentences.append(sent)
            continue
        words = sent.split()
        seen = set()
        new_words = []

        for word in words:
            if word.lower() not in seen:
                seen.add(word.lower())
                new_words.append(word)
        cleaned_sent = " ".join(new_words)
        cleaned_sentences.append(cleaned_sent)
    return ".".join(cleaned_sentences)
def replace_consecutive_periods(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'\.(?:\s*\.)+', '.', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def remove_duplicate_sentences(text: str, case_sensitive: bool = False) -> str:
    if not text:
        return text
    ends_with_period = text.rstrip().endswith('.')
    sentences = text.split(".")

    seen = set()
    unique_sentences = []

    for sent in sentences:
        stripped = sent.strip()
        if not stripped:
            continue
        key = stripped if case_sensitive else stripped.lower()
        if key not in seen:
            seen.add(key)
            unique_sentences.append(stripped)
    result = ". ".join(unique_sentences)
    if ends_with_period and not result.endswith('.'):
        result += '.'
    return result