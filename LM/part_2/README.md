## Part 2 (11 points)
**Mandatory requirements**: For the following experiments the perplexity must be below 250 (***PPL < 250***) and it should be lower than the one achieved in Part 1.1 (i.e. base LSTM).

Starting from the `LM_RNN` in which you replaced the RNN with a LSTM model, apply the following regularisation techniques:
- Weight Tying 
- Variational Dropout (no DropConnect)
- Non-monotonically Triggered AvSGD 

These techniques are described in [this paper](https://openreview.net/pdf?id=SyyGPP0TZ).
