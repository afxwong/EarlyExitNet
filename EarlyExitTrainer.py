import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from OptionalExitModule import TrainingState
from EarlyExitGateLoss import EarlyExitGateLoss
import time
import pickle

class ModelTrainer:
    def __init__(self, model, device, model_dir = os.path.join("models"), alpha=0.5):
        self.model = model
        self.device = device
        self.model_dir = model_dir
        
        self.classifier_loss_function = nn.CrossEntropyLoss()
        self.gate_loss_function = None # gets set by set_alpha call
        self.alpha = None # gets set by set_alpha call
        
        self.writer = SummaryWriter(log_dir=os.path.join(self.model_dir, "runs"))
        self.progress_bar = None
        
        # create the gate loss function
        self.set_alpha(alpha)
        
    # MARK: - Training Classifiers
    def train_classifier_epoch(self, train_loader, epoch, validation_loader=None):
        self.model.train()
        net_loss = 0.0
        net_accuracy = 0.0
        
        validation_loss = 0.0
        validation_accuracy = 0.0
        
        self.progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', ncols=100, leave=False)
        
        for i, (X, y) in enumerate(self.progress_bar):
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
            self.progress_bar.set_postfix({"Loss": loss.item(), "Accuracy": accuracy})
        
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
            validation_accuracies = []
            validation_losses = []
            
            print("Training classifier for exit", i+1)
                  
            # set exits before you to be forward, with you being exit
            self.model.exit_modules[i].set_state(TrainingState.TRAIN_CLASSIFIER_EXIT)
            for j in range(i):
                self.model.exit_modules[j].set_state(TrainingState.TRAIN_CLASSIFIER_FORWARD)
                
            max_accuracy = 0.0
            # train the classifier
            for epoch in range(epoch_count):
                # TODO: bail out early if last 5 validation accuracies are decreasing
                loss, accuracy, validation_loss, validation_accuracy = self.train_classifier_epoch(train_loader, epoch, validation_loader)
                validation_losses.append(validation_loss)
                validation_accuracies.append(validation_accuracy)
                
                # write to tensorboard
                self.writer.add_scalar(f'Loss/train/classifier {i}', loss, epoch)
                self.writer.add_scalar(f'Accuracy/train/classifier {i}', accuracy, epoch)
                self.writer.add_scalar(f'Loss/validation/classifier {i}', validation_loss, epoch)
                self.writer.add_scalar(f'Accuracy/validation/classifier {i}', validation_accuracy, epoch)
            
                if self.should_stop_early(validation_accuracies):
                    print("Validation accuracies are decreasing, stopping training early")
                    break   
                
                if validation_accuracy > max_accuracy:
                    max_accuracy = validation_accuracy
                    self.save_model(f'exit_{i+1}_classifier.pth')
            
        # train the final classifier
        print("Training final classifier")
        validation_accuracies = []
        validation_losses = []
        max_accuracy = 0.0
        for i in range(len(self.model.exit_modules)):
            self.model.exit_modules[i].set_state(TrainingState.TRAIN_CLASSIFIER_FORWARD)
            
        for epoch in range(epoch_count):
            # TODO: bail out early if last 5 validation accuracies are decreasing
            loss, accuracy, validation_loss, validation_accuracy = self.train_classifier_epoch(train_loader, epoch, validation_loader)
            validation_losses.append(validation_loss)
            validation_accuracies.append(validation_accuracy)
            
            # write to tensorboard
            self.writer.add_scalar(f'Loss/train/classifier {len(self.model.exit_modules)}', loss, epoch)
            self.writer.add_scalar(f'Accuracy/train/classifier {len(self.model.exit_modules)}', accuracy, epoch)
            self.writer.add_scalar(f'Loss/validation/classifier {len(self.model.exit_modules)}', validation_loss, epoch)
            self.writer.add_scalar(f'Accuracy/validation/classifier {len(self.model.exit_modules)}', validation_accuracy, epoch)
            
            if self.should_stop_early(validation_accuracies):
                print("Validation accuracies are decreasing, stopping training early")
                break
            
            if validation_accuracy > max_accuracy:
                max_accuracy = validation_accuracy    
                self.save_model(f'final_classifier.pth')
        self.writer.flush()
           
    # MARK: - Training Exits
    def train_exit_epoch(self, train_loader, lr, epoch, validation_loader=None):
        self.model.train()
        
        net_loss = 0.0
        validation_accuracy = 0.0
        validation_time = 0.0
        
        self.progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', ncols=100, leave=False)
        
        for i, (X, y) in enumerate(self.progress_bar):
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
            
            optimizer = torch.optim.Adam(trainable_params, lr=lr)
            
            optimizer.zero_grad()
            
            loss, ce_part, cost_part = self.gate_loss_function(y, y_hats, exit_confidences, self.model.costs)
            
            net_loss += loss.item()

            loss.backward()
            optimizer.step()
            
            # Update and display the progress bar at the end of each epoch
            self.progress_bar.set_postfix({"Loss": loss.item()})
                
            # Optionally, calculate validation metrics
            if validation_loader is not None and i % (len(train_loader) // 10) == 0:
                validation_accuracy, validation_time, exit_idx = self.validate_exit_gates(validation_loader)
              
                # write to tensorboard
                self.writer.add_scalar(f'Loss/alpha={self.alpha}/exit gates', loss, (epoch * len(train_loader) + i))
                self.writer.add_scalar(f'Loss Part 1/alpha={self.alpha}/exit gates', ce_part.item(), (epoch * len(train_loader) + i))
                self.writer.add_scalar(f'Loss Part 2/alpha={self.alpha}/exit gates', cost_part.item(), (epoch * len(train_loader) + i))

                

                self.writer.add_scalar(f'Accuracy/alpha={self.alpha}/exit gates', validation_accuracy, (epoch * len(train_loader) + i))
                self.writer.add_scalar(f'Time/alpha={self.alpha}/exit gates', validation_time, (epoch * len(train_loader) + i))
                self.writer.add_scalar(f'Exit Idx/alpha={self.alpha}/exit gates', exit_idx, (epoch * len(train_loader) + i))
                self.model.train()
                self.model.set_state(TrainingState.TRAIN_EXIT)
            
        return net_loss / len(train_loader), validation_accuracy, validation_time, exit_idx
            
    def train_exit_layers(self, train_loader, lr, epoch_count=1, validation_loader=None):
        validation_accuracies = []
        validation_times = []
        exit_idx_runs = []
        max_accuracy = 0.0
        
        for epoch in range(epoch_count):
            # run the train cycle
            
            # set model state
            self.model.set_state(TrainingState.TRAIN_EXIT)
            loss, validation_accuracy, validation_time, exit_idx = self.train_exit_epoch(train_loader, lr, epoch, validation_loader)
            validation_accuracies.append(validation_accuracy)
            validation_times.append(validation_time)
            exit_idx_runs.append(exit_idx)

            print(f'Epoch {epoch} Validation Time {validation_time}')
            print(f'Epoch {epoch} Validation Accuracy {validation_accuracy}')
            print("=====================================")
            
            
            if self.should_stop_early(validation_accuracies):
                print("Validation accuracies are decreasing, stopping training early")
                break
            
            if validation_accuracy > max_accuracy:
                max_accuracy = validation_accuracy
                # save the model
                alpha_without_decimals = str(self.alpha).replace('.', '_')
                self.save_model(f'full_model_with_exit_gates_alpha_{alpha_without_decimals}.pth')
            self.writer.flush()
            
        return validation_accuracies[-1], validation_times[-1], exit_idx_runs[-1]
            
    # MARK: - Training Helpers
    def calculate_accuracy(self, y_hat, y):
        return torch.argmax(y_hat, dim=1).eq(y).sum().item() / len(y)
    
    def validate_classifier(self, validation_loader):
        self.model.eval()
        # purposefully do not change the state of the model since we are only validating the classifier
        
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
        total_exit_index_taken = 0.0
        
        with torch.no_grad():
            for i, (X_val, y_val) in enumerate(validation_loader):
                starttime = time.time()
                X_val = X_val.to(self.device)
                y_val = y_val.to(self.device)
                y_hat_val = self.model(X_val)
                
                totaltime = time.time() - starttime
                val_accuracy = self.calculate_accuracy(y_hat_val, y_val)

                exits_taken_count = torch.tensor(self.model.num_exits_per_module, device=self.device)
                weights = torch.arange(len(exits_taken_count), device=self.device) + 1

                weighted_avg = (torch.sum(exits_taken_count * weights)).item() / len(X_val)
                if self.progress_bar is not None:
                    self.progress_bar.set_postfix({"Accuracy": val_accuracy, "Time": totaltime, 
                                               "Avg Exit Idx": weighted_avg})
                
                total_accuracy += val_accuracy
                total_time += totaltime
                total_exit_index_taken += weighted_avg
                
                
        return total_accuracy / len(validation_loader), total_time / len(validation_loader), total_exit_index_taken / len(validation_loader)
    
    # MARK: - Utils
    def set_alpha(self, alpha):
        self.alpha = alpha
        self.gate_loss_function = EarlyExitGateLoss(self.device, alpha)
        
    
    def should_stop_early(self, validation_accuracy_list):
        if len(validation_accuracy_list) < 5:
            return False
        
        # return true if we are above 99% accuracy
        if validation_accuracy_list[-1] > 0.99:
            return True
        
        # return true if the last validation accuracies are decreasing
        if validation_accuracy_list[-1] < validation_accuracy_list[-2] < validation_accuracy_list[-3]: 
            return True
        return False
    
    def save_model(self, model_name):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, model_name))
        
    def load_model(self, model_name):
        self.model = pickle.load(open(os.path.join(self.model_dir, model_name), 'rb'))
