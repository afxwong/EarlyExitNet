import torch 
import torch.nn as nn
import torchvision.models as models
from EarlyExitModel import EarlyExitModel, TrainingState
from EarlyStopException import EarlyExitException
import os
from model_architectures import VGG, DenseNet, ResNet
from args import model_names, dataset_names
import logging

class ModelLoader:
    
    def __init__(self, args, device, dataloader=None):        
        self.model_type = args.arch
        self.dataset = args.data
        self.model_dir = args.save_path
        self.alpha = args.alpha
        self.use_pretrained_arch = args.use_pretrained_arch
        
        self.dataloader = dataloader
        self.device = device
 
    def load_model(self, num_outputs, trained_classifiers=False, pretrained=False):
        assert not (trained_classifiers and pretrained), "Cannot have both trained and untrained gate layers"
        
        if pretrained or trained_classifiers:
            if pretrained:
                assert self.alpha is not None, "Alpha must be specified if pretrained is True"
            assert self.dataloader is not None, "Dataloader must be specified if loading prior model"
        
        should_add_layers = pretrained or trained_classifiers
        if self.model_type == 'resnet50':
            model = self.load_resnet50(num_outputs, should_add_layers)
        elif self.model_type == 'vgg11_bn':
            model = self.load_vgg11(num_outputs, should_add_layers)
        elif self.model_type == 'densenet121':
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
            state_dict_path = os.path.join(self.model_dir, model_name)   
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
       
    def load_resnet50(self, num_outputs, pretrained=False):
        logging.info(f"Loading EarlyExit ResNet50 model architecture...")
        resnet = ResNet.ResNet50(num_classes=num_outputs, pretrained=self.use_pretrained_arch, dataset=self.dataset)
        
        if not self.use_pretrained_arch:
            # set requires_grad to False to freeze the parameters
            for param in resnet.parameters():
                param.requires_grad = False
                
        model = EarlyExitModel(resnet, num_outputs, self.device)
        model.clear_exits()
        model.set_state(TrainingState.TRAIN_CLASSIFIER_FORWARD)
        
        self.add_exits(model, ['layer1', 'layer2', 'layer3'], pretrained)
        
        return model
    
    def load_vgg11(self, num_outputs, pretrained=False):
        logging.info(f"Loading EarlyExit VGG11 model architecture...")
        vggModel = VGG.vgg11_bn(pretrained=self.use_pretrained_arch, dataset=self.dataset)
        if not self.use_pretrained_arch:
            # set requires_grad to False to freeze the parameters
            for param in vggModel.parameters():
                param.requires_grad = False
        model = EarlyExitModel(vggModel, num_outputs, self.device)
        model.clear_exits()
        model.set_state(TrainingState.TRAIN_CLASSIFIER_FORWARD)
        self.add_exits(model, ['features.8', 'features.15', 'features.22', 'avgpool'], pretrained)
        
        return model
    
    def load_densenet_cifar100(self, num_outputs, pretrained=False):
        logging.info(f"Loading EarlyExit DenseNet121 model architecture...")
        densenet = DenseNet.densenet121(num_classes=num_outputs, pretrained=self.args.use_pretrained_arch, dataset=self.args.data)
        if not self.args.use_pretrained_arch:
            # set requires_grad to False to freeze the parameters
            for param in densenet.parameters():
                param.requires_grad = False
        model = EarlyExitModel(densenet, num_outputs, self.device)
        model.clear_exits()
        
        model.set_state(TrainingState.TRAIN_CLASSIFIER_FORWARD)
        self.add_exits(model, ['dense2', 'dense3', 'dense4'], pretrained)
        
        return model