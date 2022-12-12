FROM python:3.8.3

# Maintainer info
LABEL maintainer="mahdi.austian@gmail.com"

# Make working directories
RUN  mkdir -p  /monreader-app
WORKDIR  /monreader-app

# Upgrade pip with no cache
RUN pip install --no-cache-dir -U pip

# Copy application requirements file to the created working directory
COPY requirements.txt .

# Install application dependencies from the requirements file
RUN pip install -r requirements.txt

# Copy every file in the source folder to the created working directory
COPY  . .

EXPOSE 5000

# Run the python application
CMD ["python", "main.py"]