import torch 
import torch.nn as nn
import torchvision.models as models
from EarlyExitModel import EarlyExitModel, TrainingState
from EarlyStopException import EarlyExitException
import os
from model_architectures import VGG_CIFAR10, DenseNet
from args import model_names
import logging

class ModelLoader:
    
    def __init__(self, args, device, alpha=None, dataloader=None):        
        self.validate_model_type(args.arch) # ensure correct model type
        self.model_type = args.arch
        self.dataset = args.data
        self.dataloader = dataloader
        self.alpha = alpha
        self.device = device
        
    def validate_model_type(self, model_type):
        if model_type not in model_names:
            raise Exception("Model type {} not supported.".format(model_type))
        
    def load_model(self, num_outputs, trained_classifiers=False, pretrained=False):
        assert not (trained_classifiers and pretrained), "Cannot have both trained and untrained gate layers"
        
        if pretrained or trained_classifiers:
            if pretrained:
                assert self.alpha is not None, "Alpha must be specified if pretrained is True"
            assert self.dataloader is not None, "Dataloader must be specified if loading prior model"
        
        should_add_layers = pretrained or trained_classifiers
        if self.model_type == 'resnet50':
            model = self.load_resnet(num_outputs, should_add_layers)
        elif self.model_type == 'vgg11_bn' and self.dataset == 'cifar10':
            model = self.load_vgg_cifar10(num_outputs, should_add_layers)
        elif self.model_type == 'vgg11_bn' and self.dataset == 'cifar100':
            model = self.load_vgg_cifar100(num_outputs, should_add_layers)
        elif self.model_type == 'dense121':
            model = self.load_densenet_cifar100(num_outputs, should_add_layers)
        else:
            raise Exception("Model type {} not supported.".format(self.model_type))
            
        # load prior model state if needed
        if pretrained or trained_classifiers:
            logging.debug("Setting model weights...")
            if pretrained:
                alpha_no_decimals = str(self.alpha).replace('.', '_')
                model_name = f"full_model_with_exit_gates_alpha_{alpha_no_decimals}.pth"
            else:
                model_name = f"final_classifier.pth"
            state_dict_path = os.path.join('models', self.model_type, model_name)   
            assert os.path.exists(state_dict_path), f"State dict path {state_dict_path} does not exist"
            model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
            
        # reset the states
        model.set_state(TrainingState.INFER)
            
        model.to(self.device)
        return model
    
    def add_exits(self, model, exit_layer_attrs, should_add_layers):
        logging.debug("Adding exits...")
        if should_add_layers:
            X, _ = next(iter(self.dataloader))
        
        for layer in exit_layer_attrs:
            model.add_exit(layer)
        
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
            except EarlyExitException: pass
       
    def load_resnet(self, num_outputs, pretrained=False):
        logging.info(f"Loading EarlyExit ResNet50 model architecture...")
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
        logging.info(f"Loading EarlyExit VGG11 model architecture...")
        vggModel = VGG_CIFAR10.vgg11_bn(pretrained=True)
        # set requires_grad to False to freeze the parameters
        for param in vggModel.parameters():
            param.requires_grad = False
        vggModel.classifier[-1] = nn.Linear(vggModel.classifier[-1].in_features, num_outputs)
        
        model = EarlyExitModel(vggModel, num_outputs, self.device)
        model.clear_exits()
        model.set_state(TrainingState.TRAIN_CLASSIFIER_FORWARD)
        self.add_exits(model, ['features.8', 'features.15', 'features.22', 'avgpool'], pretrained)
        
        return model
            
            
    def load_vgg_cifar100(self, num_outputs, pretrained=False):
        logging.info(f"Loading EarlyExit VGG11 model architecture...")
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
    
    def load_densenet_cifar100(self, num_outputs, pretrained=False):
        logging.info(f"Loading EarlyExit DenseNet121 model architecture...")
        densenet = DenseNet.densenet121(num_classes=num_outputs, pretrained=True)
        # set requires_grad to False to freeze the parameters
        for param in densenet.parameters():
            param.requires_grad = False
        densenet.linear = nn.Linear(densenet.linear.in_features, num_outputs)
        model = EarlyExitModel(densenet, num_outputs, self.device)
        model.clear_exits()
        
        model.set_state(TrainingState.TRAIN_CLASSIFIER_FORWARD)
        self.add_exits(model, ['dense2', 'dense3', 'dense4'], pretrained)
        
        return model