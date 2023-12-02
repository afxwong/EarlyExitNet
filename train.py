import torch
import torch.nn as nn
import logging
from args import arg_parser, modify_args
from DataLoader import CustomDataLoader
from ModelLoader import ModelLoader
from EarlyExitTrainer import ModelTrainer

class Runner:
    
    def __init__(self, args):
        logging.basicConfig(level = args.log_level, format = '%(asctime)s - %(levelname)s - %(message)s')
        self.args = modify_args(args)
        self.trainer = None
        self.find_torch_device()
        
    def find_torch_device(self):
        logging.debug(f"PyTorch version: {torch.__version__}")
        if not self.args.use_gpu:
            self.device = "cpu"
            logging.info(f"Using device: {self.device}")
            return

        # Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
        logging.debug(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
        logging.debug(f"Is MPS available? {torch.backends.mps.is_available()}")

        # Check for CUDA support
        logging.debug(f"Is CUDA available? {torch.cuda.is_available()}")

        # Set the device
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        logging.info(f"Using device: {self.device}")
        
    def get_dataloader(self):
        self.dataloader = CustomDataLoader(args=self.args)
        self.train_dataloader, self.test_dataloader, _ = self.dataloader.get_dataset(
            self.args.data, self.args.batch_size, self.args.test_batch_size)
        
    def get_model(self, trained_classifiers=False, pretrained=False):
        self.modelloader = ModelLoader(self.args, self.device, self.test_dataloader)
        self.model = self.modelloader.load_model(self.args.num_classes, trained_classifiers, pretrained)
        
    def train_classifiers(self):
        self.get_model()
        
        self.trainer = ModelTrainer(self.model, self.device, self.args.save_path)
        self.trainer.train_classifiers(self.train_dataloader, epoch_count=self.args.classifier_epochs, validation_loader=self.test_dataloader)
        
    def train_gates(self):
        if self.args.alpha is None:
            logging.warning("Alpha not specified. Training alpha range.")
            alpha_range = [i / 100 for i in range(0, 101, 5)]
        else:
            alpha_range = [self.args.alpha]
            
        for alpha in alpha_range:
            logging.info(f"Training alpha = {alpha}")
            self.args.alpha = alpha
            self.get_model(trained_classifiers=True)
            if self.trainer is None:
                self.trainer = ModelTrainer(self.model, self.device, self.args.save_path)
            else:
                self.trainer.model = self.model
            self.trainer.set_alpha(alpha)
            
            self.trainer.train_exit_layers(self.train_dataloader, lr=self.args.lr, epoch_count=self.args.gate_epochs, validation_loader=self.test_dataloader)
        
        
if __name__ == "__main__":
    runner = Runner(arg_parser.parse_args())
    runner.get_dataloader()
    # runner.train_classifiers()
    runner.train_gates()
        
