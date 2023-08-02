FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt requirements.txt

# Update package lists and install required libraries for open-cv
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the Streamlit app files to the container
COPY . .

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "Main.py", "--server.port", "8501", "--server.fileWatcherType", "none"]
