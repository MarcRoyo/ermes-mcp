FROM python:3.14-slim

# Install system dependencies
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

# Upgrade pip and install Poetry
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir poetry

# Create non-root application user
RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:/home/user/app/.venv/bin:$PATH \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true

WORKDIR /home/user/app

# Create application directories and assign ownership
RUN mkdir -p \
        /home/user/app/data \
        /home/user/app/cache \
    && chown -R user:user /home/user/app

USER user

# Copy dependency files first to improve Docker layer caching
COPY --chown=user:user pyproject.toml poetry.lock ./

# Install runtime dependencies
RUN poetry install \
    --only main \
    --no-root

# Copy application source
COPY --chown=user:user . .

EXPOSE 8182

CMD ["python", "-u", "MCPCustomServer.py"]