# Part 1 (4 points)
As for LM project, you have to apply these two modifications incrementally. Also in this case you may have to play with the hyperparameters and optimizers to improve the performance.

Modify the baseline architecture Model by:
- Adding bidirectionality
- Adding dropout layer

**Intent classification**: accuracy < br >
**Slot filling**: F1 score with conll

***Dataset to use: ATIS***
The dataset we are using is the ATIS, a collection of records of questions asked by passengers about flights in the US.

We have 4978 train samples (that will be subdivided in train/eval) and 893 test samples:

> TRAIN size: 4480  
> DEV size: 498  
> TEST size: 893  

- Intent classification is evaluated with `accuracy`
- Slot filling is evaluating with `F1 score` (at the chunk level)
