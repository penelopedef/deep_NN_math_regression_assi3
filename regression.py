import torch
from torch import nn


def create_linear_regression_model(input_size, output_size):
    """
    Create a linear regression model with the given input and output sizes.
    Hint: use nn.Linear
    """
    model = nn.Linear(input_size, output_size)
    return model


def train_iteration(X, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def fit_regression_model(X, y):
    """
    Train the model for the given number of epochs.
    Hint: use the train_iteration function.
    Hint 2: while woring you can use the print function to print the loss every 1000 epochs.
    Hint 3: you can use the previos_loss variable to stop the training when the loss is not changing much.
    """
    learning_rate = 0.001 # Pick a better learning rate
    num_epochs = 1000 # Pick a better number of epochs
    input_features = X.shape[1] # extract the number of features from the input `shape` of X
    output_features = y.shape[1] # extract the number of features from the output `shape` of y
    model = create_linear_regression_model(input_features, output_features)
    
    loss_fn = nn.MSELoss() # Use mean squared error loss, like in class

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    previous_loss = float("inf")

    '''for epoch in range(1, num_epochs):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        if False: # Change this condition to stop the training when the loss is not changing much.
            break
    '''
    for epoch in range(1, num_epochs + 1):  # Start from epoch 1
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        
        if epoch % 100 == 0:  # Print the loss every 100 epochs
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}')

        # Check if the loss is not changing much
        if abs(previous_loss - loss.item()) < 1e-5:
            print(f'Converged at epoch {epoch}, Loss: {loss.item()}')
            break
        
        previous_loss = loss.item()
        # This is a good place to print the loss every 1000 epochs.
    return model, loss

