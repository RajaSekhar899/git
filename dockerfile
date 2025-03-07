# Use an official Python runtime as a parent image

FROM python:3.11.4

# Set the working directory to /app

WORKDIR /app

# Copy the current directory contents into the container at /appC:\Users\Navat\OneDrive\Desktop\ML\saving_model

COPY . /app

# Install any needed packages specified in requirements.txt

RUN pip install -r requirements.txt


# Make port 5000 available to the world outside this container

EXPOSE 5000


# Define environment variable

ENV NAME World


# Run app.py when the container launches

CMD ["python", "app.py"]

