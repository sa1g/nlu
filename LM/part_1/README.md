# Mandatory Exam Exercise
## Part 1 (4 points)
In this, you have to modify the baseline LM_RNN by adding a set of techniques that might improve the performance. In this, you have to add one modification at a time incrementally. If adding a modification decreases the performance, you can remove it and move forward with the others. However, in the report, you have to provide and comment on this unsuccessful experiment.  For each of your experiments, you have to print the performance expressed with Perplexity (PPL).
<br>
One of the important tasks of training a neural network is  hyperparameter optimization. Thus, you have to play with the hyperparameters to minimise the PPL and thus print the results achieved with the best configuration (in particular <b>the learning rate</b>). 
These are two links to the state-of-the-art papers which use vanilla RNN [paper1](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5947611), [paper2](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf). 

**Mandatory requirements**: For the following experiments the perplexity must be below 250 (***PPL < 250***).

1. Replace RNN with a Long-Short Term Memory (LSTM) network --> [link](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
2. Add two dropout layers: --> [link](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
    - one after the embedding layer, 
    - one before theV last linear layer
3. Replace SGD with AdamW --> [link](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)




--------------------------

If you are using Colab, run these lines:  
`!wget -P dataset/PennTreeBank https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.test.txt`  
`!wget -P dataset/PennTreeBank https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.valid.txt`  
`!wget -P dataset/PennTreeBank https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.train.txt`  

Run this if you are on Colab  
`!python -m spacy download en_core_web_lg`  

--------------------------

- `utils.py`: you have to put all the functions needed to preprocess and load the dataset
- `model.py`: the class of the model defined in PyTorch.
- `functions.py`: you have to write all the other required functions (taken from the notebook and/or written on your own) needed to complete the exercise.
- `main.py`: you have to write the calls to the functions needed to output the results asked by the exercise.
- `README.md`: you may want to write a message for us related to your solution (optional).
- `/dataset`: the files of that dataset that you used.
- `/bin`: the binary files of the best models that you have trained.