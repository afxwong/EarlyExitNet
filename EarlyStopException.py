class EarlyExitException(Exception):
    def __init__(self, message="all samples have early exited"):
        super().__init__(message)