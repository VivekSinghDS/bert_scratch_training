import torch
from tqdm.auto import tqdm
from pathlib import Path
from transformers import RobertaTokenizer
import pickle 

paths = [str(x) for x in Path('./cleaned_data').glob('**/*.txt')]
print(len(paths))
print(paths[:5])
tokenizer = RobertaTokenizer.from_pretrained('./retail_tokenized_data/', max_len = 512)
input_ids = []
mask = []
labels = []


def mlm(tensor):
    rand = torch.rand(tensor.shape)
    mask_arr = (rand<0.15)*(tensor>2)
    for i in range(tensor.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        tensor[i, selection] = 4

    return tensor


for path in tqdm(paths):
    with open(path, 'r', encoding = 'utf-8') as f:
        lines = f.read()

    sample = tokenizer(lines, max_length = 512, padding = 'max_length', truncation = True, return_tensors = 'pt')
    labels.append(sample.input_ids)
    mask.append(sample.attention_mask)
    input_ids.append(mlm(sample.input_ids.detach().clone()))


input_ids = torch.cat(input_ids)
mask = torch.cat(mask)
labels = torch.cat(labels)
filename = 'input_ids_pickle'
filename2 = 'mask_pickle'
filename3 = 'labels_pickle'


with open(filename, 'wb') as f:
    pickle.dump(input_ids, f)


with open(filename2, 'wb') as f:

    pickle.dump(mask, f)


with open(filename3, 'wb') as f:
    pickle.dump(labels, f)
