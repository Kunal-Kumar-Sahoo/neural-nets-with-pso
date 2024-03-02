import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_val_split(dataset, val_split=0.2):
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    return torch.utils.data.random_split(dataset, [n_train, n_val])

def train(model, criterion, optimizer, train_loader, val_loader, max_epochs, lr_scheduler=None):
    train_losses, val_losses = [], []

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
            epoch_val_loss = running_loss / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)

        if lr_scheduler:
            lr_scheduler.step(epoch_val_loss)

        print(f'Epoch [{epoch+1}/{max_epochs}], ',
              f'Train Loss: {epoch_train_loss:.4f}, ',
              f'Validation Loss: {epoch_val_loss:.4f}')
    
    return train_losses, val_losses