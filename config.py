class Config:
    def __init__(self):
        self.training_params = {
            "cifar10": {
                "resnet56": {
                    "batch_size": 64,
                    "num_epoch": 150,
                    "lr": 0.1,
                    "lr_type": "multistep",
                    "decay_rate": 0.1,
                    "decay_epochs": [90, 130],
                    "weight_decay": 5e-4,
                    "momentum": 0.9,
                    "optimizer": "sgd",
                },
                "vgg11_bn": {
                    "batch_size": 64,
                    "num_epoch": 150,
                    "lr": 0.1,
                    "lr_type": "multistep",
                    "decay_rate": 0.1,
                    "decay_epochs": [50, 100],
                    "weight_decay": 5e-4,
                    "momentum": 0.9,
                    "optimizer": "sgd",
                },
            },
            "cifar100": {
                "densenet121": {
                    "batch_size": 128,
                    "num_epoch": 150,
                    "lr": 0.1,
                    "lr_type": "multistep",
                    "decay_rate": 0.1,
                    "decay_epochs": [50, 100],
                    "weight_decay": 5e-4,
                    "momentum": 0.9,
                    "optimizer": "sgd",
                },
                "vgg11_bn": {
                    "batch_size": 64,
                    "num_epoch": 150,
                    "lr": 0.1,
                    "lr_type": "multistep",
                    "decay_rate": 0.1,
                    "decay_epochs": [50, 100],
                    "weight_decay": 5e-4,
                    "momentum": 0.9,
                    "optimizer": "sgd",
                },
            },
            "imagenette": {
                "resnet50": {
                    "batch_size": 64,
                    "num_epoch": 150,
                    "lr": 0.1,
                    "lr_type": "multistep",
                    "decay_rate": 0.1,
                    "decay_epochs": [90, 130],
                    "weight_decay": 1e-4,
                    "momentum": 0.9,
                    "optimizer": "sgd",
                }
            },
        }
        self.train_classifier_params = {
            "cifar10": {
                "resnet56": {
                    "batch_size": 32,
                    "num_epoch": 60,
                    "lr": 0.0001,
                    "optimizer": "adam",
                },
                "vgg11_bn": {
                    "batch_size": 64,
                    "num_epoch": 60,
                    "lr": 0.0001,
                    "optimizer": "adam",
                },
            },
            "cifar100": {
                "densenet121": {
                    "batch_size": 64,
                    "num_epoch": 60,
                    "lr": 0.0001,
                    "optimizer": "adam",
                },
                "vgg11_bn": {
                    "batch_size": 64,
                    "num_epoch": 60,
                    "lr": 0.0001,
                    "optimizer": "adam",
                },
            },
            "imagenette": {
                "resnet50": {
                    "batch_size": 32,
                    "num_epoch": 60,
                    "lr": 0.0001,
                    "optimizer": "adam",
                }
            },
        }
        self.model_params = {
            "cifar10": {
                "resnet56": {
                    "ee_layer_locations": ["layer1", "layer2", "layer3"],
                },
                "vgg11_bn": {
                    "ee_layer_locations": [
                        "features.8",
                        "features.15",
                        "features.22",
                        "avgpool",
                    ],
                },
            },
            "cifar100": {
                "densenet121": {
                    "ee_layer_locations": ["dense2", "dense3", "dense4"],
                },
                "vgg11_bn": {
                    "ee_layer_locations": [
                        "features.8",
                        "features.15",
                        "features.22",
                        "avgpool",
                    ],
                },
            },
            "imagenette": {
                "resnet56": {
                    "ee_layer_locations": ["layer1", "layer2", "layer3"],
                },
            },
        }
        self.inference_params = {
            "cifar10": {
                "resnet56": {
                    "batch_size": 32,
                    "num_epoch": 3,
                    "lr": 0.00000001,
                    "optimizer": "adam",
                },
                "vgg11_bn": {
                    "batch_size": 64,
                    "num_epoch": 60,
                    "lr": 0.000001,
                    "optimizer": "adam",
                },
            },
            "cifar100": {
                "densenet121": {
                    "batch_size": 64,
                    "num_epoch": 60,
                    "lr": 0.000001,
                    "optimizer": "adam",
                },
                "vgg11_bn": {
                    "batch_size": 64,
                    "num_epoch": 60,
                    "lr": 0.000001,
                    "optimizer": "adam",
                },
            },
            "imagenette": {
                "resnet50": {
                    "batch_size": 32,
                    "num_epoch": 60,
                    "lr": 0.00000001,
                    "optimizer": "adam",
                }
            },
        }
