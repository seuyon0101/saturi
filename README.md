# AIFFELTHON Final Project Repository 
# `saturi translator`
- #### last update : MAR 31, 2023
---
## Team name : Saturi
## Participants :

| name | role |
|------|-----|
|1. SY| (lead)|
|2. SH| (model)
|3. SA| (data)|
|4. JI| (model)|
|5. DS| (coordinator)|
|6. HW| (coordinator)|

# Description :
- ### Eng-to-Korean dialect translation(5 different dialect)
- ### We trained our model using kor-dialect corpus data downloaded from AIHUB(aihub.or.kr), and paired with their english corpus by translation using hugging face NMT checkpoint.
- ### Number of data pair for training : kangwon 300k, jeju 200k, jd 170k, cc 110k, gs 110k.
</br>

# How to use :
- ### download checkpoint -> please email me if you want to try(seuyon0101@gmail.com).
- ### python main.py
- ### insert tokenizer type
- ### vocab size small? Yes(8k vocab) , No(16k vocab)
- ### add region tag before input text

# Example :

- basic transformer model inference :

![example1](img/t1.png)
![example2](img/t2.png)
![example3](img/t3.png)
![example4](img/t4.png)


