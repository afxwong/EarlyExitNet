import torch
from torch import nn
from tqdm import tqdm
import os
from OptionalExitModule import TrainingState
from EarlyExitGateLoss import EarlyExitGateLoss
import time

class ModelTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.classifier_loss_function = nn.CrossEntropyLoss()
        self.gate_loss_function = EarlyExitGateLoss()
        
    # MARK: - Training Classifiers
    def train_classifier_epoch(self, train_loader, epoch, validation_loader=None):
        self.model.train()
        net_loss = 0.0
        net_accuracy = 0.0
        
        validation_loss = None
        validation_accuracy = None
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', ncols=100, leave=False)
        
        for i, (X, y) in enumerate(progress_bar):
            X = X.to(self.device)
            y = y.to(self.device)
            y_hat = self.model(X)
            
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = torch.optim.Adam(trainable_params, lr=0.0001)
            
            optimizer.zero_grad()
            
            loss = self.classifier_loss_function(y_hat, y)
            accuracy = self.calculate_accuracy(y_hat, y)
            
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
            validation_loss, validation_accuracy = self.validate_classifier(validation_loader)
            print(f'Epoch {epoch} Validation Loss {validation_loss}')
            print(f'Epoch {epoch} Validation Accuracy {validation_accuracy}')
            print("=====================================")
            
        return net_loss / len(train_loader), net_accuracy / len(train_loader), validation_loss, validation_accuracy
        
        
    def train_classifiers(self, train_loader, epoch_count=1, validation_loader=None):
        
        self.model.set_state(TrainingState.TRAIN_CLASSIFIER_FORWARD)
        for i in range(len(self.model.exit_modules)):
            print("Training classifier for exit", i+1)
                  
            # set exits before you to be forward, with you being exit
            self.model.exit_modules[i].set_state(TrainingState.TRAIN_CLASSIFIER_EXIT)
            for j in range(i):
                self.model.exit_modules[j].set_state(TrainingState.TRAIN_CLASSIFIER_FORWARD)
                
            # train the classifier
            for epoch in range(epoch_count):
                # TODO: bail out early if last 5 validation accuracies are decreasing
                loss, accuracy, validation_loss, validation_accuracy = self.train_classifier_epoch(train_loader, epoch, validation_loader)
                
        # train the final classifier
        print("Training final classifier")
        for i in range(len(self.model.exit_modules)):
            self.model.exit_modules[i].set_state(TrainingState.TRAIN_CLASSIFIER_FORWARD)
            
        for epoch in range(epoch_count):
            # TODO: bail out early if last 5 validation accuracies are decreasing
            loss, accuracy, validation_loss, validation_accuracy = self.train_classifier_epoch(train_loader, epoch, validation_loader)
           
    # MARK: - Training Exits
    def train_exit_epoch(self, train_loader, epoch, validation_loader=None):
        self.model.train()
        
        net_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', ncols=100, leave=False)
        
        for i, (X, y) in enumerate(progress_bar):
            X = X.to(self.device)
            y = y.to(self.device)
            y_hats, exit_confidences = self.model(X)
            
            trainable_params = None
            # concatenate all trainable parameters as gate layers
            for exit_layer in self.model.exit_modules:
                if trainable_params is None:
                    trainable_params = list(filter(lambda p: p.requires_grad, exit_layer.exit_gate.parameters()))
                else:
                    trainable_params += list(filter(lambda p: p.requires_grad, exit_layer.exit_gate.parameters()))
            
            optimizer = torch.optim.Adam(trainable_params, lr=0.0001)
            
            optimizer.zero_grad()
            
            loss = self.gate_loss_function(y, y_hats, exit_confidences, range(len(self.model.exit_modules)+1))
            
            net_loss += loss.item()

            loss.backward()
            optimizer.step()
            
            # print(self.model.exit_modules[0].exit_gate.weight)
            
            # Update and display the progress bar at the end of each epoch
            progress_bar.set_postfix({"Loss": loss.item()})
                
         # Optionally, calculate validation metrics
        if validation_loader is not None:
            validation_accuracy, validation_time = self.validate_exit_gates(validation_loader)
            print(f'Epoch {epoch} Validation Time {validation_time}')
            print(f'Epoch {epoch} Validation Accuracy {validation_accuracy}')
            print("=====================================")
            
        return net_loss / len(train_loader), validation_accuracy, validation_time
            
    def train_exit_layers(self, train_loader, epoch_count=1, validation_loader=None):
        for epoch in range(epoch_count):
            # run the train cycle
            
            # set model state
            self.model.set_state(TrainingState.TRAIN_EXIT)
            loss, validation_accuracy, validation_time = self.train_exit_epoch(train_loader, epoch, validation_loader)
            
    # MARK: - Training Helpers
    def calculate_accuracy(self, y_hat, y):
        return torch.argmax(y_hat, dim=1).eq(y).sum().item() / len(y)
    
    def validate_classifier(self, validation_loader):
        self.model.eval()
        
        # purposefully do not change the state of the model
        
        total_loss = 0.0
        total_accuracy = 0.0
        with torch.no_grad():
            for X_val, y_val in validation_loader:
                X_val = X_val.to(self.device)
                y_val = y_val.to(self.device)
                y_hat_val = self.model(X_val)
                
                val_loss = self.classifier_loss_function(y_hat_val, y_val)
                val_accuracy = self.calculate_accuracy(y_hat_val, y_val)
                total_loss += val_loss.item()
                total_accuracy += val_accuracy
        return total_loss / len(validation_loader), total_accuracy / len(validation_loader)
    
    def validate_exit_gates(self, validation_loader):
        self.model.eval()
        
        # set the model state so forward propagation works properly
        self.model.set_state(TrainingState.INFER)
        
        total_accuracy = 0.0
        total_time = 0.0
        
        with torch.no_grad():
            for X_val, y_val in validation_loader:
                starttime = time.time()
                X_val = X_val.to(self.device)
                y_val = y_val.to(self.device)
                y_hat_val = self.model(X_val)
                
                totaltime = time.time() - starttime
                val_accuracy = self.calculate_accuracy(y_hat_val, y_val)
                
                total_accuracy += val_accuracy
                total_time += totaltime
                
        return total_accuracy / len(validation_loader), total_time / len(validation_loader)
