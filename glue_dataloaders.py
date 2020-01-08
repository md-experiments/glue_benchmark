def load_BERT_data(paths, tokenizer, BATCH_SIZE=128,seed=1234):
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

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  LABEL.build_vocab(trn_data)
  trn_iter, val_iter, tst_iter = data.Iterator.splits(  #BucketIterator
      (trn_data, val_data, tst_data), 
      #sort_key=lambda x: len(x.text),
      batch_size = BATCH_SIZE, 
      device = device)


  return trn_iter, val_iter, tst_iter, LABEL, TEXT
