import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
import io
import os
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from pyvi import ViTokenizer
from model import Seq2SeqTransformer,generate_square_subsequent_mask,create_mask
from torch import nn
import torch
import numpy as np
from nltk.translate.bleu_score import  corpus_bleu
import time
from timeit import default_timer as timer
#to switch to en to vi, change the order of filepaths, place en to first and vi to second, also switch src and tgt language, tokenizer
#data path
train_filepaths=[
    r'/kaggle/input/machine-translation-dataset/PhoMT/detokenization/train/train.vi',
    r'/kaggle/input/machine-translation-dataset/PhoMT/detokenization/train/train.en'
]
val_filepaths=[
    r'/kaggle/input/machine-translation-dataset/PhoMT/detokenization/dev/dev.vi',
    r'/kaggle/input/machine-translation-dataset/PhoMT/detokenization/dev/dev.en'
]
test_filepaths=[
    r'/kaggle/input/machine-translation-dataset/PhoMT/detokenization/test/test.vi',
    r'/kaggle/input/machine-translation-dataset/PhoMT/detokenization/test/test.en'
]

#Parameter
BATCH_SIZE=64
lower=True
SRC_LANGUAGE = 'vi'
TGT_LANGUAGE = 'en'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_model = True
save_model = True

# Training hyperparameters
num_epochs = 20
learning_rate = 3e-4

def vi_tokenizer(sentence):
    tok_trans=ViTokenizer.tokenize(sentence).split()
    result=[]
    for tok in tok_trans:
        result.append(tok.replace('_',' '))
    return result

# Place-holders
token_transform = {}
vocab_transform = {}
token_transform[SRC_LANGUAGE] = vi_tokenizer
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
#Build vocab
class Field:
    def __init__(self, datapath, language,min_freq):
        self.datapath=datapath
        self.ln=language
        self.min_freq=min_freq

    def build_vocab(self):
        with io.open(self.datapath, encoding="utf8") as f:
            counter = Counter()
            for string_ in f:
                list_tok=token_transform[self.ln](string_)
                if lower:
                    list_tok=[tok.lower() for tok in list_tok]
                counter.update(list_tok)
                
        result_vocab=vocab(counter,self.min_freq,specials=['<unk>', '<pad>', '<bos>', '<eos>'])
        result_vocab.set_default_index(UNK_IDX)
        print(f'build vocab {self.ln} successfully!')
        return result_vocab
    


vietnamese=Field(train_filepaths[0],SRC_LANGUAGE,min_freq=10)
english=Field(train_filepaths[1],TGT_LANGUAGE, min_freq=10)
vocab_transform[SRC_LANGUAGE]=vietnamese.build_vocab()
vocab_transform[TGT_LANGUAGE]=english.build_vocab()
print('vi vocab size', len(vocab_transform[SRC_LANGUAGE]))
print('en vocab size', len(vocab_transform[TGT_LANGUAGE]))

#data transform
# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids): # type: ignore
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))
def lower_transform(token_ids):
    if lower:
        return [tok.lower() for tok in token_ids]
    else:
        return token_ids
# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               lower_transform,
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor

class MTDataset(Dataset):
    def __init__(self, datapath) -> None:
        self.data=self.data_process(datapath)
    def data_process(self,filepaths):
        raw_src_iter = iter(io.open(filepaths[0], encoding="utf8"))
        raw_tgt_iter = iter(io.open(filepaths[1], encoding="utf8"))
        data = []
        for (raw_src, raw_tgt) in zip(raw_src_iter, raw_tgt_iter):
            raw_src,raw_tgt=raw_src.strip(),raw_tgt.strip()
            vi_tensor_ = text_transform[SRC_LANGUAGE](raw_src)
            en_tensor_ = text_transform[TGT_LANGUAGE](raw_tgt)
            if len(en_tensor_)>64 or len(vi_tensor_)>64:
                continue
            data.append((vi_tensor_, en_tensor_))
        return data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
train_data=MTDataset(train_filepaths)
val_data=MTDataset(val_filepaths)
test_data=MTDataset(test_filepaths) 

#Dataloader

# function to collate data samples into batch tensors
def generate_batch(batch):
    src_batch=[sentence[0] for sentence in batch]
    tgt_batch=[sentence[1] for sentence in batch]
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE,
                       shuffle=True, collate_fn=generate_batch)


#model implementation
torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 256
NHEAD = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM).to(DEVICE)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)



optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)
loss_fn  = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

#load model
model_path='/kaggle/input/transformer-epoch1/pytorch/rework/1/rework.pth'
if os.path.exists(model_path) and load_model:
    state_dict=torch.load(model_path,map_location=DEVICE)
    model.load_state_dict(state_dict)
else:
    print('no dict found!')

#train
def train(model, optimizer):
    model.train()
    losses = 0

    for batch_idx, (src,tgt) in enumerate(train_dataloader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    mean_loss = losses / (batch_idx+1)
    return mean_loss
def evaluate(model):
    model.eval()
    losses=0
    for batch_idx, (src,tgt) in enumerate(val_dataloader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / (batch_idx+1)


max_loss=float('inf')
acception=0
for epoch in range(1, num_epochs+1):
    start_time = timer()
    train_loss = train(model, optimizer)
    scheduler.step(train_loss)
    end_time = timer()
    val_loss = evaluate(model)
    scheduler.step(train_loss)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    
    
    if val_loss>max_loss:
        acception+=1
        if acception>3:
            print('overfitting caution!')
            break
    else:
        acception=0
        max_loss=val_loss
        if save_model:
            # Define the path where you want to save the model
            model_save_path = f'transformer_model_{epoch}.pth'

            # Save the model's state dictionary
            torch.save(model.state_dict(), model_save_path)




def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 10, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

print(translate(model, "Thế giới ngoài kia không biết họ yêu nhau kiểu gì ?"))

#calculate bleu score
def calculate_bleu_score(model, src_sentence, ref_sentence):
    """
    Calculates the BLEU score for a given source sentence and its reference translation.
    
    Args:
    - model (torch.nn.Module): The translation model.
    - src_sentence (str): The source sentence.
    - ref_sentence (str): The reference sentence for evaluation.

    Returns:
    - float: The BLEU score.
    """
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 10, start_symbol=BOS_IDX).flatten()
    translated_tokens = vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))
    translated_tokens.remove('<bos>')
    if '<eos>' in translated_tokens:
        translated_tokens.remove('<eos>')
    # Tokenize sentences
    ref_tokens = token_transform[TGT_LANGUAGE](ref_sentence)
    ref_tokens=[tok.lower() for tok in ref_tokens]
    # Calculate BLEU score
    return [ref_tokens], translated_tokens


st=time.time()
raw_src_iter = iter(io.open(test_filepaths[0], encoding="utf8"))
raw_tgt_iter = iter(io.open(test_filepaths[1], encoding="utf8"))
total=0
refls=[]
predls=[]
for idx,(raw_src, raw_tgt) in enumerate(zip(raw_src_iter, raw_tgt_iter)):
    raw_src,raw_tgt=raw_src.strip(),raw_tgt.strip()
    ref,pred= calculate_bleu_score(model,raw_src,raw_tgt)
    refls.append(ref)
    predls.append(pred)
print('bleu score is:',corpus_bleu(refls,predls))
print('calculated time:', time.time()-st)