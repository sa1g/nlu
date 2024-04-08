# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import torch
import math


def train_loop(data: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.modules.loss._Loss, model: torch.nn.Module, clip:int=5) -> float:
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad()
        
        output = model(sample["source"])
        
        loss = criterion(output, sample["target"])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        
        # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        loss.backward()  
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
    return sum(loss_array) / sum(number_of_tokens)


def eval_loop(data:torch.utils.data.DataLoader, eval_criterion: torch.nn.modules.loss._Loss, model: torch.nn.Module):
    model.eval()
    
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad():  # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample["source"])
            loss = eval_criterion(output, sample["target"])
            
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    
    return ppl, loss_to_return


def init_weights(mat):
    """
    Initializes the weights of the given module using specific initialization methods.

    Args:
        mat (nn.Module): The module for which the weights need to be initialized.

    Returns:
        None
    """

    for m in mat.modules():
        if type(m) in [torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN]:
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(
                            param[idx * mul : (idx + 1) * mul]
                        )
                elif "weight_hh" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul : (idx + 1) * mul])
                elif "bias" in name:
                    param.data.fill_(0)
        else:
            if type(m) in [torch.nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
