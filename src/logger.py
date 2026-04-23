import logging            # Used to track events (info, errors, etc.)
import os                 # Used for file and folder operations
from datetime import datetime   # Used to get current date and time


# Create a unique log file name using current date and time
# Example: 04_23_2026_23_45_10.log
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"


# Create path for "logs" folder inside current working directory
# Example: your_project/logs
logs_dir = os.path.join(os.getcwd(), "logs")


# Create the "logs" folder if it doesn't already exist
# exist_ok=True → avoids error if folder is already present
os.makedirs(logs_dir, exist_ok=True)


# Create full path for the log file
# Example: your_project/logs/04_23_2026_23_45_10.log
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)


# Configure logging system
logging.basicConfig(
    filename=LOG_FILE_PATH,   # File where logs will be stored
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    # Format explains:
    # asctime → time of log
    # lineno → line number where log is written
    # name → module name
    # levelname → type of log (INFO, ERROR, etc.)
    # message → actual log message

    level=logging.INFO,  # Log only INFO level and above (INFO, WARNING, ERROR)
)

