## Part 2 (11 points)
**Mandatory requirements**: For the following experiments the perplexity must be below 250 (***PPL < 250***) and it should be lower than the one achieved in Part 1.1 (i.e. base LSTM).

Starting from the `LM_RNN` in which you replaced the RNN with a LSTM model, apply the following regularisation techniques:
- 4.4 Weight Tying
    - Weight tying (Inan et al., 2016; Press & Wolf, 2016) shares the weights between the embedding
and softmax layer, substantially reducing the total parameter count in the model. The technique has
theoretical motivation (Inan et al., 2016) and prevents the model from having to learn a one-to-one
correspondence between the input and output, resulting in substantial improvements to the standard
LSTM language model.
- 4.2 Variational Dropout (no DropConnect)
    - In standard dropout, a new binary dropout mask is sampled each and every time the dropout function
is called. New dropout masks are sampled even if the given connection is repeated, such as the input
x0 to an LSTM at timestep t = 0 receiving a different dropout mask than the input x1 fed to the
same LSTM at t = 1. A variant of this, variational dropout (Gal & Ghahramani, 2016), samples
a binary dropout mask only once upon the first call and then to repeatedly use that locked dropout
mask for all repeated connections within the forward and backward pass.
While we propose using DropConnect rather than variational dropout to regularize the hidden-to-
hidden transition within an RNN, we use variational dropout for all other dropout operations, specif-
ically using the same dropout mask for all inputs and outputs of the LSTM within a given forward
and backward pass. Each example within the minibatch uses a unique dropout mask, rather than a
single dropout mask being used over all examples, ensuring diversity in the elements dropped out.
- 3 Non-monotonically Triggered AvSGD ( reducing the risk of overfitting and eventually leading to a better estimate for our model parameters.we only trigger the averaging step when a certain validation metric fails to improve for multiple cycles. This "non-monotonic" criterion ensures that the randomness of training does not play a major role in the decision to average, resulting in a better estimate for our parameters. The conservative nature of the criterion also ensures that we only take the averaging step when we are certain of its necessity, further reducing the risk of overfitting and leading to a more generalizable model.
)


These techniques are described in [this paper](https://openreview.net/pdf?id=SyyGPP0TZ).
