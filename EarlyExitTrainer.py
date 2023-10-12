import torch
from EarlyExitLoss import EarlyExitWeightedLoss
from tqdm import tqdm

class ModelTrainer:
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
        try:
            # Clear the cache to prevent memory leaks
            torch.cuda.empty_cache()
        except:
            pass   
        self.model.train()
        
        net_loss = 0.0
        net_accuracy = 0.0
        
        validation_loss = None
        validation_accuracy = None
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', ncols=100, leave=False)
        
        for i, (X, y) in enumerate(progress_bar):
            X = X.to(self.device)
            y = y.to(self.device)
            y_hat, exit_points, gate_logits = self.model(X)
            
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = torch.optim.Adam(trainable_params, lr=0.001)
            
            optimizer.zero_grad()
            
            # Get only the classes that are in the dataset, since the model outputs 1000 classes and data is 
            shortened_yhat = y_hat #self.calculate_proper_classes(y_hat)
            
            loss = self.calculate_loss(shortened_yhat, y, exit_points, gate_logits, shouldWeight=shouldWeight)
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
            validation_loss, validation_accuracy, _ = self.validate(validation_loader)
            print(f'Epoch {epoch} Validation Loss {validation_loss}')
            print(f'Epoch {epoch} Validation Accuracy {validation_accuracy}')
            
        return net_loss / len(train_loader), net_accuracy / len(train_loader), validation_loss, validation_accuracy
                
    def train(self, train_loader, epoch_count=1, validation_loader=None):
       
        # iterate through epochs for each classification head
        for layer_idx, layer in enumerate(self.model.exit_modules):
            losses = []
            accuracies = []
        
            # reset the model
            for tmp_layer_idx, tmp_layer in enumerate(self.model.exit_modules):
                tmp_layer.remove_forces()
                
            print(f'Training early exit layer {layer_idx+1}')
            layer.force_exit()
           
            for epoch in range(epoch_count):
                # bail out early if last three validation losses are increasing and last three validation accuracies are decreasing
                if len(losses) > 20 and len(accuracies) > 20:
                    if losses[-1] > losses[-2] > losses[-3] and accuracies[-1] < accuracies[-3]:
                        print('Model is overfitting, stopping early')
                        break
                
                print(f'Beginning epoch {epoch}')
                loss, acc, val_loss, val_acc = self.train_epoch(train_loader, epoch, validation_loader, shouldWeight=False)
                
                losses.append(val_loss)
                accuracies.append(val_acc)
        
        # now we need to train the last classifier head
        losses = []
        accuracies = []
        for layer in self.model.exit_modules:
            layer.remove_forces()
            layer.force_forward()
            
        for epoch in range(epoch_count):
            # bail out early if last three validation losses are increasing and last three validation accuracies are decreasing
            if len(losses) > 20 and len(accuracies) > 20:
                if losses[-1] > losses[-2] > losses[-3] and accuracies[-1] < accuracies[-3]:
                    print('Model is overfitting, stopping early')
                    break
                
            print(f'Beginning epoch {epoch} on final classifier head')
            loss, acc, val_loss, val_acc = self.train_epoch(train_loader, epoch, validation_loader, shouldWeight=False)
                
            losses.append(val_loss)
            accuracies.append(val_acc)
        
        
        # now we need to train the full model
        losses = []
        accuracies = []
        for layer in self.model.exit_modules:
            layer.remove_forces()
            layer.freeze_classifier()
            
        for epoch in range(epoch_count):
            # bail out early if last three validation losses are increasing and last three validation accuracies are decreasing
            if len(losses) > 20 and len(accuracies) > 20:
                if losses[-1] > losses[-2] > losses[-3] and accuracies[-1] < accuracies[-3]:
                    print('Model is overfitting, stopping early')
                    break
                
            print(f'Beginning epoch {epoch} with no forced exits')
            loss, acc, validation_loss, validation_accuracy = self.train_epoch(train_loader, epoch, validation_loader, shouldWeight=True)
            
            losses.append(validation_loss)
            accuracies.append(validation_accuracy)
            
            # save the model
            torch.save(self.model.state_dict(), f'models/model_{epoch}.pt')
               
    def calculate_proper_classes(self, y_hat):
        # TODO: replace this with transfer learning on resnet to get the proper classes
        relevant_indices = list(self.map.keys())
        
        # Index into the y_hat tensor to get the relevant values
        yhat_relevant = y_hat[:, relevant_indices]
        return yhat_relevant
            

    def calculate_loss(self, y_hat, y, exit_points, confidences, shouldWeight=True):
        return self.loss_function(y_hat, y, exit_points, confidences, shouldWeight=shouldWeight)
    
    def calculate_accuracy(self, y_hat, y):
        return torch.argmax(y_hat, dim=1).eq(y).sum().item() / len(y)

    def validate(self, validation_loader):
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_exit_points = 0.0
        with torch.no_grad():
            for X_val, y_val in validation_loader:
                X_val = X_val.to(self.device)
                y_val = y_val.to(self.device)
                y_hat_val, exit_points, gate_logits = self.model(X_val)
                
                y_hat_val_shortened = y_hat_val #self.calculate_proper_classes(y_hat_val)
                
                val_loss = self.calculate_loss(y_hat_val_shortened, y_val, exit_points, gate_logits)
                val_accuracy = self.calculate_accuracy(y_hat_val_shortened, y_val)
                total_loss += val_loss.item()
                total_accuracy += val_accuracy
                total_exit_points += exit_points.mean().item()
        return total_loss / len(validation_loader), total_accuracy / len(validation_loader), total_exit_points / len(validation_loader)
