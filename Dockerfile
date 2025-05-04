FROM python:3.9

WORKDIR /app

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy project files into the container
COPY . /app/

# Command to run your script
CMD ["python", "main.py"]
