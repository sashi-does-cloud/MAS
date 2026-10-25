import os

START_SEQUENCE = "\033[90m"
END_SEQUENCE = "\033[0m"
WARNING_SEQUENCE = "\033[93m"
ERROR_SEQUENCE = "\033[91m"

def log(msg):
    if os.getenv("DEBUG") == "True":
        print(f" {START_SEQUENCE} {msg} {END_SEQUENCE}")


def warn(msg):
    if os.getenv("DEBUG") == "True":
        print(f" {WARNING_SEQUENCE} {msg} {END_SEQUENCE}")

def error(msg):
    if os.getenv("DEBUG") == "True":
        print(f" {ERROR_SEQUENCE} {msg} {END_SEQUENCE}")
    

