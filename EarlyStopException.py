class EarlyExitException(Exception):
    
    def __init__(self, message="all samples have early exited", y_hat=None):
        self.y_hat = y_hat
        super().__init__(message)