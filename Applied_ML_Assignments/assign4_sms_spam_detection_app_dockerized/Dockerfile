# start from an official python image
FROM python:3.9-slim

# set the working directory in the container
WORKDIR /app

# copy the requirements file into the container and install the dependencies
COPY requirements.txt /app/

# install dependencies using pip
RUN pip install -r requirements.txt

# copy the source code into the container
COPY src/ /app/src/
# this also copies the main app.py and score.py files into the container

# expose port 5000 for the Flask app to listen on
EXPOSE 5000

# run the command to start the Flask app
CMD ["python", "app/src/app.py"]