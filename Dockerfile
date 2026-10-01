# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Install required packages for building eSpeak and general utilities
RUN apt-get update && apt-get install -y \
        build-essential \
        autoconf \
        automake \
        libtool \
        pkg-config \ 
        git \
        cmake \ 
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip 

RUN mkdir -p cache && chmod 777 cache

RUN useradd -m -u 1000 user

USER user

RUN mkdir -p /home/user/app/data

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app
# Onnx install 

COPY --chown=user requirements.txt $HOME/app/

RUN pip install -r requirements.txt

COPY --chown=user . $HOME/app/

# Fix ownership issues
USER root
RUN chown -R user:user $HOME/app
USER user

EXPOSE 8182

CMD ["python3", "-u", "MCPCustomServer.py"]


