from torchtext import data
import torch

from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd

def df_tsv(path, header=True, columns_dict={}):
  a=[]
  with open(path, 'r') as f:
    for i in f:
      a+=[i[:-1].split('\t')]
  if header:
    res=pd.DataFrame(a[1:], columns=a[0])
  else:
    res=pd.DataFrame(a)
  
  if len(columns_dict)>0:
    res.columns=list(columns_dict.keys())

    res=res.rename(columns=columns_dict)


    res=res.copy().drop('', axis=1, errors='ignore')

  return res

def prep_inputs_masks(df, category_dict, tokenizer, task=0, MAX_LEN=128):
  samp_id =2
  
  # Create sentence and label lists
  sentences = df.sent.values

  labels = df.label.apply(lambda x: float(category_dict[x])).values


  # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
  if task =='sentiment':
    sentences = ["[CLS] " + sentences[idx]+ " [SEP]"  for idx in range(len(sentences))]
    #sentences = ["[CLS] " + sentences[idx] + " [SEP]" + ' ' +str(mentions[idx]) + " [SEP]" for idx in range(len(sentences))]
  elif task =='similarity':
    sentences1 = df.sent1.values
    sentences = ["[CLS] " + sentences[idx] + " [SEP] " + sentences1[idx] + " [SEP]" for idx in range(len(sentences))]
    #sentences = [("[CLS] " + sentences[idx].lower() + " [SEP]").replace(str(mentions[idx].lower()),' [MASK] ') + ' '+ str(mentions[idx]) + " [SEP]"  \
    #             for idx in range(len(sentences))]
  else:
    print('BOOM: wrong task')
  tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
  
  # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
  input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
  # Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway. 
  # In the original paper, the authors used a length of 512.
  MAX_LEN=MAX_LEN
  # Pad our input tokens
  input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

  return input_ids, labels

def gen_DataLoader(train_inputs, train_labels, batch_size = 32):
  train_data = TensorDataset(train_inputs, train_labels)
  train_sampler = RandomSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size,pin_memory=True,num_workers=4)
  return train_dataloader

def datagen_BERT(path, header, columns_dict, category_dict, tokenizer, task, batch_size, max_len=128):

  df_train=df_tsv(path, header=header, columns_dict=columns_dict)

  train_inputs, train_labels=prep_inputs_masks(df_train, category_dict, tokenizer, task=task, MAX_LEN=max_len)
  train_inputs = torch.tensor(train_inputs)
  train_labels = torch.tensor(train_labels)
  train_dataloader=gen_DataLoader(train_inputs, train_labels, batch_size = batch_size)
  return train_dataloader, train_inputs, df_train



def load_BERT_data(paths, tokenizer, task_type, device, BATCH_SIZE=128,seed=1234):
  init_token_idx = tokenizer.cls_token_id
  eos_token_idx = tokenizer.sep_token_id
  pad_token_idx = tokenizer.pad_token_id
  unk_token_idx = tokenizer.unk_token_id

  max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

  def tokenize_and_cut(sentence):
      tokens = tokenizer.tokenize(sentence) 
      tokens = tokens[:max_input_length-2]
      return tokens
  
  TEXT = data.Field(batch_first = True,
                    use_vocab = False,
                    tokenize = tokenize_and_cut,
                    preprocessing = tokenizer.convert_tokens_to_ids,
                    init_token = init_token_idx,
                    eos_token = eos_token_idx,
                    pad_token = pad_token_idx,
                    unk_token = unk_token_idx)

  LABEL = data.LabelField(dtype = torch.float)
  
  for key in paths.keys():

    for i,label_value in enumerate(paths[key]['schema']):
      if label_value[1]=='LABEL':
        paths[key]['schema'][i]=(label_value[0],LABEL)
      elif label_value[1]=='TEXT':
        paths[key]['schema'][i]=(label_value[0],TEXT)

  trn_path=paths['train']['path']
  trn_schema=paths['train']['schema']
  try:
    val_path=paths['validation']['path']
    val_schema=paths['validation']['schema']
    val_split=True
  except:
    val_split=False
  tst_path=paths['test']['path']
  tst_schema=paths['test']['schema']

  if val_split:
    trn_data=data.TabularDataset(trn_path,'tsv', fields=trn_schema)
    val_data=data.TabularDataset(val_path,'tsv', fields=val_schema)
    tst_data=data.TabularDataset(tst_path,'tsv', fields=tst_schema)
  else:
    trn_data, val_data = data.TabularDataset(trn_path,'tsv', fields=trn_schema)
    tst_data = data.TabularDataset(tst_path,'tsv', fields=tst_schema)

  #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  assert(device==torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


  LABEL.build_vocab(trn_data)
  trn_iter, val_iter, tst_iter = data.Iterator.splits(  #BucketIterator
      (trn_data, val_data, tst_data), 
      #sort_key=lambda x: len(x.text),
      sort=False,
      batch_size = BATCH_SIZE, 
      device = device)


  return trn_iter, val_iter, tst_iter, LABEL, TEXT
