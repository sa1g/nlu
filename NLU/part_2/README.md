## Part 2 (11 points)

Adapt the code to fine-tune a pre-trained BERT model using a multi-task learning setting on intent classification and slot filling. 
You can refer to this paper to have a better understanding of how to implement this: https://arxiv.org/abs/1902.10909. In this, one of the challenges of this is to handle the sub-tokenization issue.

*Note*: The fine-tuning process is to further train on a specific task/s a model that has been pre-trained on a different (potentially unrelated) task/s.


The models that you can experiment with are [*BERT-base* or *BERT-large*](https://huggingface.co/google-bert/bert-base-uncased). 

**Intent classification**: accuracy <br>
**Slot filling**: F1 score with conll

***Dataset to use: ATIS***