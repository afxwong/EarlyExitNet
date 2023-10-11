import torch
from EarlyExitLoss import EarlyExitWeightedLoss
from tqdm import tqdm

class EarlyExitTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.loss_function = EarlyExitWeightedLoss(total_exit_points=len(model.exit_modules)+1)
        
        # create map for output -> proper class
        self.map = {
            0: 0,
            217: 1,
            482: 2,
            491: 3,
            497: 4,
            566: 5,
            569: 6,
            571: 7,
            574: 8,
            701: 9,
            
        }

    def train_epoch(self, train_loader, epoch, validation_loader=None, shouldWeight=True):
        self.model.train()
        
        net_loss = 0.0
        net_accuracy = 0.0
        
        validation_loss = None
        validation_accuracy = None
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', ncols=100, leave=False)
        
        for i, (X, y) in enumerate(progress_bar):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            X = X.to(self.device)
            y = y.to(self.device)
            
            optimizer.zero_grad()
            y_hat, exit_points = self.model(X)
            
            # Get only the classes that are in the dataset, since the model outputs 1000 classes and data is 
            shortened_yhat = self.calculate_proper_classes(y_hat)
            
            loss = self.calculate_loss(shortened_yhat, y, exit_points, shouldWeight=shouldWeight)
            accuracy = self.calculate_accuracy(shortened_yhat, y)
            
            net_loss += loss.item()
            net_accuracy += accuracy

            loss.backward()
            optimizer.step()
            
            # Update and display the progress bar at the end of each epoch
            progress_bar.set_postfix({"Loss": loss.item(), "Accuracy": accuracy})
        
        print(f'Epoch {epoch} Loss {net_loss / len(train_loader)}')
        print(f'Epoch {epoch} Accuracy {net_accuracy / len(train_loader)}')
            

        # Optionally, calculate validation loss
        if validation_loader is not None:
            validation_loss, validation_accuracy = self.validate(validation_loader)
            print(f'Epoch {epoch} Validation Loss {validation_loss}')
            print(f'Epoch {epoch} Validation Accuracy {validation_accuracy}')
            
        progress_bar.close()
            
        return net_loss / len(train_loader), net_accuracy / len(train_loader), validation_loss, validation_accuracy
                
    def train(self, train_loader, epoch_count=1, validation_loader=None):
        # iterate through epochs for each classification head
        for layer_idx, layer in enumerate(self.model.exit_modules):
            
            # reset the model
            for tmp_layer_idx, tmp_layer in enumerate(self.model.exit_modules):
                tmp_layer.remove_forces()
                
            print(f'Training early exit layer {layer_idx+1}')
            layer.force_exit()
           
            for epoch in range(epoch_count):
                print(f'Beginning epoch {epoch}')
                self.train_epoch(train_loader, epoch, shouldWeight=False)
        
        # now we need to train only the gate layers
        for layer in self.model.exit_modules:
            layer.remove_forces()
            
        for epoch in range(epoch_count):
            print(f'Beginning epoch {epoch} with no forced exits')
            
            test_loader = validation_loader if epoch % 5 == 0 else None
            loss, accuracy, validation_loss, validation_accuracy = self.train_epoch(train_loader, epoch, test_loader, shouldWeight=True)
            
            for layer in self.model.exit_modules:
                print(f'Layer exit gate weights: {layer.exit_gate.weight}')
            
        
               
    def calculate_proper_classes(self, y_hat):
        # TODO: replace this with transfer learning on resnet to get the proper classes
        relevant_indices = list(self.map.keys())
        
        # Index into the y_hat tensor to get the relevant values
        yhat_relevant = y_hat[:, relevant_indices]
        return yhat_relevant
            

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
                
                y_hat_val_shortened = self.calculate_proper_classes(y_hat_val)
                
                val_loss = self.calculate_loss(y_hat_val_shortened, y_val, exit_points)
                val_accuracy = self.calculate_accuracy(y_hat_val_shortened, y_val)
                total_loss += val_loss.item()
                total_accuracy += val_accuracy
        return total_loss / len(validation_loader), total_accuracy / len(validation_loader)
