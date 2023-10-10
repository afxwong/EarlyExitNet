import torch
from EarlyExitLoss import EarlyExitWeightedLoss

class EarlyExitTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.loss_function = EarlyExitWeightedLoss(total_exit_points=len(model.exit_modules)+1)
        
        # create map for output -> proper class
        self.map = {}

    def train_epoch(self, train_loader, optimizer, epoch, validation_loader=None, shouldWeight=True):
        self.model.train()
        for i, (X, y) in enumerate(train_loader):
            X = X.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()
            y_hat, exit_points = self.model(X)
            loss = self.calculate_loss(y_hat, y, exit_points, shouldWeight=False)

            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch} Batch {i} Loss {loss.item()}')
            print(f'Epoch {epoch} Batch {i} accuracy {self.calculate_accuracy(y_hat, y)}')
            

        # Optionally, calculate validation loss
        if validation_loader is not None:
            validation_loss, validation_accuracy = self.validate(validation_loader)
            print(f'Epoch {epoch} Validation Loss {validation_loss}')
            print(f'Epoch {epoch} Validation Accuracy {validation_accuracy}')
                
    def train(self, train_loader, optimizer, epoch_count, validation_loader=None):
        # iterate through epochs for each classification head
        for layer_idx, layer in enumerate(self.model.exit_modules):
            
            # reset the model
            for tmp_layer_idx, tmp_layer in enumerate(self.model.exit_modules):
                tmp_layer.force_forward(False)
                tmp_layer.force_exit(False)
                
            print(f'Training early exit layer {layer}')
            layer.force_forward(False)
            layer.force_exit()
           
            for epoch in range(epoch_count):
                print(f'Beginning epoch {epoch}')
                self.train_epoch(train_loader, optimizer, epoch, shouldWeight=False)
                
        # now we need to train the last layer by forcing all other layers to continue
        for layer in self.model.exit_modules:
            layer.force_forward()
            layer.force_exit(False)
        
        for epoch in range(epoch_count):
            print(f'Beginning epoch {epoch} with no forced exits')
            self.train_epoch(train_loader, optimizer, epoch, validation_loader, shouldWeight=False)
        
        
        # now we need to train the entire model with nothing forced
        for layer in self.model.exit_modules:
            layer.force_forward(False)
            layer.force_exit(False)
            
        for epoch in range(epoch_count):
            print(f'Beginning epoch {epoch} with no forced exits')
            self.train_epoch(train_loader, optimizer, epoch, validation_loader, shouldWeight=True)
        
               
    # def calculate_proper_classes(self, y_hat):
            

    def calculate_loss(self, y_hat, y, exit_points, shouldWeight=True):
        # TODO: calculate loss
        return self.loss_function(y_hat, y, exit_points, shouldWeight=shouldWeight)
    
    def calculate_accuracy(self, y_hat, y):
        return torch.argmax(y_hat, dim=1).eq(y).sum().item() / len(y)

    def validate(self, validation_loader):
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        with torch.no_grad():
            for X_val, y_val in validation_loader:
                X_val = X_val.to(self.device)
                y_val = y_val.to(self.device)
                y_hat_val, exit_points = self.model(X_val)
                val_loss = self.calculate_loss(y_hat_val, y_val, exit_points)
                val_accuracy = self.calculate_accuracy(y_hat_val, y_val)
                total_loss += val_loss.item()
                total_accuracy += val_accuracy
        return total_loss / len(validation_loader), total_accuracy / len(validation_loader)
