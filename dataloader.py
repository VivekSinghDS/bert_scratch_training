
import torch
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import AdamW
import pickle
from tqdm.auto import tqdm

with open('input_ids_pickle', 'rb') as f:
    input_ids = pickle.load(f)

with open('labels_pickle', 'rb') as f:
    label = pickle.load(f)

with open('mask_pickle', 'rb') as f:
    mask= pickle.load(f)
encodings = {
    'input_ids':input_ids, 
    'attention_mask':mask,
    'labels':label
}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        return {key:tensor[i] for key, tensor in self.encodings.items()}

dataset = Dataset(encodings)

dataloader = torch.utils.data.DataLoader(dataset, batch_size = 16, shuffle = True)

config = RobertaConfig(
    vocab_size = 1000000, 
    max_position_embedding = 514, 
    hidden_size = 768, 
    num_attention_heads = 12,
    num_hidden_layers = 6, 
    type_vocab_size = 1
)

model = RobertaForMaskedLM(config)
device = torch.device('cuda')if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optim = AdamW(model.parameters(), lr = 1e-5)
epochs = 2

for epoch in range(epochs):
    loop = tqdm(dataloader, leave = True)
    for batch in loop:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask = mask, labels = labels)

        loss = outputs.loss
        loss.backward()
        optim.step()
        loop.set_description(f'Epoch : {epoch}')
        loop.set_postfix(loss = loss.item())


model.save_pretrained('./retail_model')

