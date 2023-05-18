import pandas as pd
import re
import sentencepiece as spm


# save to pickle

def save_pickle_data(src_valid_corpus,tgt_valid_corpus,enc_corpus_test,dec_corpus_test,full_data_test, name='test') :
    test_dict = dict()
    test_dict['toks_en'] = src_valid_corpus
    test_dict['toks_dec'] = tgt_valid_corpus
    test_dict['source_txt'] = enc_corpus_test
    test_dict['target_txt'] = dec_corpus_test
    test_dict['topic'] = full_data_test.topic.values
    test_dict['reg'] = full_data_test.reg.values
    dtest= pd.DataFrame(test_dict)
    dtest.to_pickle(data_dir + f'/data_{name}.pkl','gzip')

# preprocess

def preprocess_sentence(sentence):

    sentence = sentence.lower()
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!<>가-힣ㄱ-ㅎㅏ-ㅣ]+", " ", sentence)
    sentence = sentence.strip()

    return sentence


# sentencepiece tokenizer

# Sentencepiece를 활용하여 학습한 tokenizer를 생성
def generate_tokenizer(corpus, vocab_size, lang="en", pad_id=0, bos_id=1, eos_id=2, unk_id=3):

    temp_file = os.getenv('HOME') + f'/aiffel/DATA/corpus_{lang}_r0.txt'     # corpus를 받아 txt파일로 저장
    
    with open(temp_file, 'w') as f:
        for row in corpus:
            f.write(str(row) + '\n')
    
    # Sentencepiece를 이용해 
    spm.SentencePieceTrainer.Train(
        f'--input={temp_file} --model_type=bpe --pad_id={pad_id} --bos_id={bos_id} --eos_id={eos_id} \
        --unk_id={unk_id} --model_prefix=spm_{lang}_v --vocab_size={vocab_size} \
        --user_defined_symbols=<jj>,<jd>,<gs>,<cc>,<kw> --remove_extra_whitespaces=false'   # model_r0
    )
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(f'spm_{lang}_v.model') # model_r0

    return tokenizer