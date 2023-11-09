import torch 
import torch.nn as nn
import torchvision.models as models
from EarlyExitModel import EarlyExitModel, TrainingState
from EarlyStopException import EarlyExitException
import os
import shutil
import pickle

class ModelLoader:
    
    def __init__(self, model_type, device, alpha=None, dataloader=None):
        self.download_cifar10()
        
        self.validate_model_type(model_type) # ensure correct model type
        self.model_type = model_type
        self.dataloader = dataloader
        self.alpha = alpha
        self.device = device
        
    def download_cifar10(self):
        # TODO: replace this 
        if not os.path.exists('cifar10_models'):
            # run command
            os.system("git clone https://github.com/huyvnphan/PyTorch_CIFAR10")
    
            # copy cifar10_models folder to current directory
            shutil.copytree(os.path.join("PyTorch_CIFAR10", "cifar10_models"), "cifar10_models")
            
            # delete the cloned repo
            try:
                shutil.rmtree("PyTorch_CIFAR10")
            except: pass
        from cifar10_models import vgg
        
    def validate_model_type(self, model_type):
        if model_type not in ["resnet", "vgg_cifar10", "vgg_cifar100"]:
            raise Exception("Model type {} not supported.".format(model_type))
        
    def load_model(self, num_outputs, trained_classifiers=False, pretrained=False):
        assert not (trained_classifiers and pretrained), "Cannot have both trained and untrained gate layers"
        
        if pretrained or trained_classifiers:
            if pretrained:
                assert self.alpha is not None, "Alpha must be specified if pretrained is True"
            assert self.dataloader is not None, "Dataloader must be specified if loading prior model"
        
        should_add_layers = pretrained or trained_classifiers
        if self.model_type == 'resnet':
            model = self.load_resnet(num_outputs, should_add_layers)
        elif self.model_type == 'vgg_cifar10':
            model = self.load_vgg_cifar10(num_outputs, should_add_layers)
        elif self.model_type == 'vgg_cifar100':
            model = self.load_vgg_cifar100(num_outputs, should_add_layers)
            
        # reset the states
        model.set_state(TrainingState.INFER)
            
        # load prior model state if needed
        if pretrained or trained_classifiers:
            print("Setting model weights...")
            if pretrained:
                alpha_no_decimals = str(self.alpha).replace('.', '_')
                model_name = f"full_model_with_exit_gates_alpha_{alpha_no_decimals}.pth"
            else:
                model_name = f"final_classifier.pth"
            state_dict_path = os.path.join('models', self.model_type, model_name)   
            assert os.path.exists(state_dict_path), f"State dict path {state_dict_path} does not exist"
            model.load_state_dict(torch.load(state_dict_path))
            
        model.to(self.device)
        return model
    
    def add_exits(self, model, exit_layer_attrs, should_add_layers):
        print("Adding exits...")
        if should_add_layers:
            X, _ = next(iter(self.dataloader))
        
        for layer in exit_layer_attrs:
            model.add_exit(layer, self.model_type)
        
            if not should_add_layers: continue
            # now since we don't know the shape of the input, we need to run a forward pass
            # this will generate the classifiers and exit gates
            # only needed to load state dict properly
            
            # set the exits to force forward except the last one
            for i in range(len(model.exit_modules) - 1):
                exit_module = model.exit_modules[i]
                exit_module.set_state(TrainingState.TRAIN_CLASSIFIER_FORWARD)  
            model.exit_modules[-1].set_state(TrainingState.TRAIN_CLASSIFIER_EXIT)
            
            # run one data batch to create classifiers with proper shape
            try:
                model.model(X)
            except EarlyExitException: 
                pass
        
       
    def load_resnet(self, num_outputs, pretrained=False):
        print(f"Loading EarlyExit ResNet50 model architecture...")
        resnet = models.resnet50(pretrained=True)
        
        # set requires_grad to False to freeze the parameters
        for param in resnet.parameters():
            param.requires_grad = False
        
        resnet.fc = nn.Linear(resnet.fc.in_features, num_outputs)
        model = EarlyExitModel(resnet, num_outputs, self.device)
        model.clear_exits()
        model.set_state(TrainingState.TRAIN_CLASSIFIER_FORWARD)
        
        self.add_exits(model, ['layer1', 'layer2', 'layer3'], pretrained)
        
        return model
    
    def load_vgg_cifar10(self, num_outputs, pretrained=False):
        print(f"Loading EarlyExit VGG11 model architecture...")
        model_path = os.path.join('models', 'vgg_cifar10', 'vgg11_bn.pkl')
        vgg = pickle.load(open(model_path, 'rb'))
        # set requires_grad to False to freeze the parameters
        for param in vgg.parameters():
            param.requires_grad = False
        vgg.classifier[-1] = nn.Linear(vgg.classifier[-1].in_features, num_outputs)
        
        model = EarlyExitModel(vgg, num_outputs, self.device)
        model.clear_exits()
        model.set_state(TrainingState.TRAIN_CLASSIFIER_FORWARD)
        self.add_exits(model, ['features.8', 'features.15', 'features.22', 'avgpool'], pretrained)
        
        return model
            
            
    def load_vgg_cifar100(self, num_outputs, pretrained=False):
        print(f"Loading EarlyExit VGG11 model architecture...")
        vgg = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=True)
        # set requires_grad to False to freeze the parameters
        for param in vgg.parameters():
            param.requires_grad = False
        vgg.classifier[-1] = nn.Linear(vgg.classifier[-1].in_features, num_outputs)
        model = EarlyExitModel(vgg, num_outputs, self.device)
        model.clear_exits()
        
        model.set_state(TrainingState.TRAIN_CLASSIFIER_FORWARD)
        self.add_exits(model, ['features.8', 'features.15', 'features.22', "classifier.0"], pretrained)
        
        return model