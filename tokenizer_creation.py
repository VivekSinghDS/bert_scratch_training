from pathlib import Path
import os
from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path('./cleaned_data').glob('**/*.txt')]
print(paths[:5])
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files = paths, vocab_size = 1000000, min_frequency = 2,
                special_tokens = ['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

os.mkdir('./retail_tokenized_data')
tokenizer.save_model('retail_tokenized_data')