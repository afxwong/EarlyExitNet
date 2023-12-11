import torch
import torch.nn as nn
import logging
from args import arg_parser, modify_args
from DataLoader import CustomDataLoader
from ModelLoader import ModelLoader
from EarlyExitTrainer import ModelTrainer
import os
from config import Config


class Runner:
    def __init__(self, args):
        self.args = modify_args(args)
        self.config = Config()
        self.trainer = None

        self.setup_logging()
        self.find_torch_device()

    def setup_logging(self):
        os.system("rm log.txt")
        # Create a logger
        self.logger = logging.getLogger("LEENet")

        # Create a FileHandler for logging to a file (warnings and higher)
        fileHandler = logging.FileHandler("log.txt")
        fileHandler.setLevel(logging.DEBUG)

        # Create a StreamHandler for logging to the console (debug and higher)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(self.args.log_level)

        # Create a formatter (customize as needed)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

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

    def get_dataloader(self, train_type):
        self.dataloader = CustomDataLoader(args=self.args)
        if train_type == "train":
            batch_size = self.config.training_params[self.args.data][self.args.arch][
                "batch_size"
            ]
        elif train_type == "classifier":
            batch_size = self.config.train_classifier_params[self.args.data][
                self.args.arch
            ]["batch_size"]
        elif train_type == "gate":
            batch_size = self.config.inference_params[self.args.data][self.args.arch][
                "batch_size"
            ]

        self.train_dataloader, self.test_dataloader, _ = self.dataloader.get_dataset(
            self.args.data, batch_size
        )

    def get_model(self, trained_classifiers=False, pretrained=False):
        self.modelloader = ModelLoader(self.args, self.device, self.test_dataloader)
        self.model = self.modelloader.load_model(
            self.args.num_classes, trained_classifiers, pretrained
        )
        
    def train_base_model(self):
        self.get_dataloader("train")
        self.get_model()
        self.trainer = ModelTrainer(self.model, self.device, self.args)
        # save to state dict folder and train full model
        self.trainer.set_model_dir(
            os.path.join("model_architectures", "state_dicts")
        )
        self.trainer.train_full_model(
            self.train_dataloader,
            epoch_count=self.config.training_params[self.args.data][self.args.arch][
                "num_epoch"
            ],
            validation_loader=self.test_dataloader,
        )
        # save to save path and train classifiers
        self.trainer.set_model_dir(self.args.save_path)

    def train_classifiers(self):
        self.get_dataloader("classifier")
        self.get_model()

        self.trainer = ModelTrainer(self.model, self.device, self.args)
        self.trainer.train_classifiers(
            self.train_dataloader,
            epoch_count=self.config.train_classifier_params[self.args.data][self.args.arch]["num_epoch"],
            validation_loader=self.test_dataloader,
        )

    def train_gates(self):
        self.get_dataloader("gate")
        if self.args.alpha is None:
            logging.warning("Alpha not specified. Training alpha range.")
            alpha_range = [i / 100 for i in range(0, 101, 5)]
        else:
            alpha_range = [self.args.alpha]

        lr = self.config.inference_params[self.args.data][self.args.arch]["lr"]
        gate_epochs = self.config.inference_params[self.args.data][self.args.arch]["num_epoch"]

        for alpha in alpha_range:
            logging.info(f"Training alpha = {alpha}")
            self.args.alpha = alpha
            self.get_model(trained_classifiers=True)
            self.trainer = ModelTrainer(self.model, self.device, self.args)
            self.trainer.set_alpha(alpha)

            self.trainer.train_exit_layers(
                self.train_dataloader,
                lr=lr,
                epoch_count=gate_epochs,
                validation_loader=self.test_dataloader,
            )
            
    def run(self):
        if not self.args.use_pretrained_arch:
            self.train_base_model()
        if not self.args.use_pretrained_classifiers:
            runner.train_classifiers()
        runner.train_gates()


if __name__ == "__main__":
    runner = Runner(arg_parser.parse_args())
    runner.run()
