import torch
import torch.nn as nn
import logging
from args import arg_parser, modify_args
from DataLoader import CustomDataLoader
from ModelLoader import ModelLoader
from EarlyExitTrainer import ModelTrainer
import os

class Runner:
    
    def __init__(self, args):
        self.args = modify_args(args)
        self.trainer = None
        
        self.setup_logging()
        self.find_torch_device()
        
    def setup_logging(self):
        os.system("rm log.txt")
        # Create a logger
        self.logger = logging.getLogger('my_logger')

        # Create a FileHandler for logging to a file (warnings and higher)
        fileHandler = logging.FileHandler("log.txt")
        fileHandler.setLevel(logging.DEBUG)

        # Create a StreamHandler for logging to the console (debug and higher)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(self.args.log_level)

        # Create a formatter (customize as needed)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Set the formatter for the handlers
        fileHandler.setFormatter(formatter)
        consoleHandler.setFormatter(formatter)

        # Add the handlers to the logger
        logging.getLogger().addHandler(fileHandler)
        logging.getLogger().addHandler(consoleHandler)
        logging.getLogger().setLevel(logging.DEBUG)
        
    def find_torch_device(self):
        logging.debug(f"PyTorch version: {torch.__version__}")
        if self.args.use_cpu:
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
        
        self.trainer = ModelTrainer(self.model, self.device, self.args)
        if not self.args.use_pretrained_arch:
            # save to state dict folder and train full model
            self.trainer.set_model_dir(os.path.join("model_architectures", "state_dicts"))
            self.trainer.train_full_model(self.train_dataloader, epoch_count=self.args.arch_epochs, validation_loader=self.test_dataloader)
            # save to save path and train classifiers
            self.trainer.set_model_dir(self.args.save_path)
        
        
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
                self.trainer = ModelTrainer(self.model, self.device, self.args)
            else:
                self.trainer.model = self.model
            self.trainer.set_alpha(alpha)
            
            self.trainer.train_exit_layers(self.train_dataloader, lr=self.args.lr, epoch_count=self.args.gate_epochs, validation_loader=self.test_dataloader)
        
        
if __name__ == "__main__":
    runner = Runner(arg_parser.parse_args())
    runner.get_dataloader()
    runner.train_classifiers()
    runner.train_gates()
        
