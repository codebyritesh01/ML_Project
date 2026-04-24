import sys
# sys module helps us get runtime error details (like traceback)
from src.logger import logging

# Function to create a detailed error message
def error_message_detail(error, error_detail: sys):
    
    # exc_info() gives (exception type, exception object, traceback)
    _, _, exc_tb = error_detail.exc_info()
    
    # Get the file name where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Create a detailed error message with file name, line number, and actual error
    error_message = "Error occured in python script name [{0}] linenumber [{1}] error message [{2}]".format(
        file_name,          # file where error happened
        exc_tb.tb_lineno,   # line number of error
        str(error)          # actual error message
    )
    
    return error_message


# Custom Exception class (your own exception)
class CustomeException(Exception):

    # Constructor method
    def __init__(self, error_message, error_detail: sys):
        
        # Call parent Exception class constructor
        super().__init__(error_message)
        
        # Store detailed error message (with file + line info)
        self.error_message = error_message_detail(
            error_message,
            error_detail=error_detail
        )

    # This method defines what will be printed when exception is shown
    def __str__(self):
        return self.error_message