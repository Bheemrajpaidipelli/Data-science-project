import logging
import os
from datetime import datetime  # Fixed typo in module name

# Create a log filename using the current timestamp.
# Changed inner double quotes to single quotes to avoid conflict with the outer f-string quotes.
log_filename = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the logs directory path.
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)  # Create the logs folder if it doesn't exist

# Create full log file path.
log_file_path = os.path.join(logs_dir, log_filename)

# Configure the logging.
logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


