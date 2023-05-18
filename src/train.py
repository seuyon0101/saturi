import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import re
import os
import io
import wandb

from tqdm import tqdm 
import random

import sentencepiece as spm
from konlpy.tag import Mecab

print(tf.__version__)
print(pd.__version__)
print(np.__version__)

# path and tokenizer name
data_dir = './data/train_test_pickle'
tokenizer_name = input() #'spm' or 'msp' or 'cmsp'
tokenizer_vocab_size = int(input()) #8000 or 16000
name_project = input()
wandb_entity = input() # name of wandb project name ; saturi

# train data load
data_train_path = data_dir+f"/data_train_{tokenizer_name}_{tokenizer_vocab_size}_0324.pkl"
data_test_path = data_dir+f"/data_test_{tokenizer_name}_{tokenizer_vocab_size}_0324.pkl"
full_data = pd.read_pickle(data_train_path, 'gzip')
full_data_test = pd.read_pickle(data_test_path, 'gzip')

# load tokenizer model
tok_path ='ckpts/tokenizer'
sizes = tokenizer_vocab_size + 9
SRC_VOCAB_SIZE = sizes
TGT_VOCAB_SIZE = sizes

enc_tokenizer = spm.SentencePieceProcessor()
enc_tokenizer.Load(tok_path+f'/spm_enc_spm{tokenizer_vocab_size}.model')

dec_tokenizer = spm.SentencePieceProcessor()
dec_tokenizer.Load(tok_path+f'/spm_dec_{tokenizer_name}{tokenizer_vocab_size}.model')

dec_tokenizer.set_encode_extra_options("bos:eos")

# set wandb config
wandb.login()
warmups = 8500
project_name = f'{name_project}_{tokenizer_name}_{tokenizer_vocab_size}'
run  = wandb.init(project = project_name ,
                 entity = wandb_entity,
                 config = {
                     'model_name':'Vanilla_Transformer',
                     'n_layers':6,
                     'd_model':512,
                     'n_heads':8,
                     'd_ff':2048,
                     'src_vocab_size':SRC_VOCAB_SIZE,
                     'tgt_vocab_size':TGT_VOCAB_SIZE,
                     'pos_len': 512,
                     'dropout':0.2,
                     'shared':True,
                     'warmups' : warmups,
                     'batch_size' : 128,
                     'epochs':1,
                     'optimizer' :'ADAM',
                     'loss' : 'SparseCategoricalCrossentropy',
                     'metric' : 'bleu'
                 })
config = wandb.config

import sys
sys.path.insert(0, './src/MODEL/')
sys.path.insert(0, './src/PRE/')
sys.path.insert(0, './src/POST/')

from vanilla_transformer import Transformer, generate_masks

# init model
transformer = Transformer(
    n_layers=config.n_layers,
    d_model=config.d_model,
    n_heads=config.n_heads,
    d_ff=config.d_ff,
    src_vocab_size=config.src_vocab_size,
    tgt_vocab_size=config.tgt_vocab_size,
    pos_len=config.pos_len,
    dropout=config.dropout,
    shared=config.shared
)

# learning rate scheduler class define
class LearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=config.warmups):
        super(LearningRateScheduler, self).__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = step ** np.array(-0.5)
        arg2 = step * np.array(self.warmup_steps ** -1.5)
        
        return np.array(self.d_model ** -0.5) * tf.math.minimum(arg1, arg2)
    
learningrate = LearningRateScheduler(512)

# optimizer
optimizer = tf.keras.optimizers.Adam(learningrate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# Loss 함수 정의
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    # Masking 되지 않은 입력의 개수로 Scaling하는 과정
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

# train function
@tf.function()
def train_step(src, tgt, model, optimizer):
    gold = tgt[:, 1:]
        
    enc_mask, dec_enc_mask, dec_mask = generate_masks(src, tgt)

    # 계산된 loss에 tf.GradientTape()를 적용해 학습을 진행합니다.
    with tf.GradientTape() as tape:
        predictions, enc_attns, dec_attns, dec_enc_attns = model(src, tgt, enc_mask, dec_enc_mask, dec_mask)
        loss = loss_function(gold, predictions[:, :-1])

    # 최종적으로 optimizer.apply_gradients()가 사용됩니다. 
    gradients = tape.gradient(loss, model.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    
    return loss, enc_attns, dec_attns, dec_enc_attns, predictions

# Validating the model
@tf.function
def model_validate(src, tgt, model):
    gold = tgt[:, 1:]
        
    enc_mask, dec_enc_mask, dec_mask = generate_masks(src, tgt)
    predictions, enc_attns, dec_attns, dec_enc_attns = model(src, tgt, enc_mask, dec_enc_mask, dec_mask)
    v_loss = loss_function(gold, predictions[:, :-1])
    
    return v_loss, predictions


# 학습 매니저 설정
from evaluation import evaluate, compute_metric, translate
full_data = full_data.sample(frac=1).copy()

def train_and_checkpoint(transformer, manager, EPOCHS):
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    
    else:
        print("Initializing from scratch.")


    for epoch in range(EPOCHS):
        
        total_loss = 0     
        
        src_corpus = full_data.loc[full_data['tok_cat']==bucket,'toks_en'].values
        tgt_corpus = full_data.loc[full_data['tok_cat']==bucket,'toks_dec'].values
        src_valid_corpus = full_data_test['toks_en'].values
        tgt_valid_corpus = full_data_test['toks_dec'].values
        max_len = full_data.loc[full_data['tok_cat']==bucket,'tok_len'].max()

        if max_len > 380 :
            batch_size = 8

        if max_len > 512 : 
            max_len = 512


        enc_train = tf.keras.preprocessing.sequence.pad_sequences(src_corpus, padding='post', maxlen=max_len)
        dec_train = tf.keras.preprocessing.sequence.pad_sequences(tgt_corpus, padding='post', maxlen=max_len)
        enc_test = tf.keras.preprocessing.sequence.pad_sequences(src_valid_corpus, padding='post', maxlen=max_len)
        dec_test = tf.keras.preprocessing.sequence.pad_sequences(tgt_valid_corpus, padding='post', maxlen=max_len)
        
        idx_list = list(range(0, enc_train.shape[0], config.batch_size))
        random.shuffle(idx_list)

        t = tqdm(idx_list)
        
        for (batch, idx) in enumerate(t):
            batch_loss, enc_attns, dec_attns, dec_enc_attns, preds = train_step(enc_train[idx:idx+batch_size],
                                                                            dec_train[idx:idx+batch_size],
                                                                            transformer,
                                                                            optimizer)

            total_loss += batch_loss

            t.set_postfix_str('Loss %.4f' % (total_loss.numpy() / (batch + 1)))
        
            wandb.log({
                    "train_loss": (total_loss.numpy() / (batch + 1)),
                    })
            
            #validation
            total_loss_val = 0
            val_size=25
            tv = tqdm(range(0,enc_test.shape[0], val_size))
            
            for (batch_val,test_idx) in enumerate(tv) :
                val_loss, val_preds = model_validate(enc_test[test_idx : test_idx + val_size],
                                          dec_test[test_idx : test_idx + val_size],
                                          transformer)
                total_loss_val += val_loss
                tv.set_postfix_str('val_Loss %.4f' % (total_loss_val.numpy() / (batch_val + 1)))
                
                wandb.log({
                           "valid_loss" : (total_loss_val.numpy() / (batch_val + 1))
                           })

            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            
directory = './ckpts/train_ckpt'
ckpt = tf.train.Checkpoint(step = tf.Variable(1), optimizer = optimizer , transformer = transformer)
manager = tf.train.CheckpointManager(ckpt, directory +f'tf_{tokenizer_name}_{tokenizer_vocab_size}',max_to_keep=6)

# run train
train_and_checkpoint(transformer, manager, config.epochs)


# metric score
from datasets import load_metric

bleu = load_metric("sacrebleu")
bleu_valid_score = []

test_text = full_data_test.eng.values
test_tgt = full_data_test.dial.values

for i in tqdm(range(len(test_text))) :
    trans = translate(test_text[i], transformer, enc_tokenizer, dec_tokenizer, verbose =False)
    label = dec_tokenizer.decode(test_tgt[i])
    result = bleu.compute(predictions=[trans], references=[[label]], smooth_method='add-k')['score']
    bleu_valid_score.append(result)

print('BLEU Score : ', np.array(bleu_valid_score).mean())