# syntax=docker/dockerfile:1

##############################################
# Frontend build stage (Node.js)
##############################################
FROM node:20-alpine AS frontend-builder

# Allows overriding the frontend directory (relative to repo root)
ARG FRONTEND_DIR=frontend
ARG FRONTEND_BUILD_DIR=dist

WORKDIR /app

# Copy the full repository so the build stage has access to shared assets
COPY . .

# Install dependencies and build the frontend only when a package.json exists
RUN if [ -f "$FRONTEND_DIR/package.json" ]; then \
        echo "Installing frontend dependencies" && \
        cd "$FRONTEND_DIR" && \
        npm install && \
        npm run build && \
        mkdir -p /opt/frontend && \
        if [ -d "$FRONTEND_BUILD_DIR" ]; then \
            cp -r "$FRONTEND_BUILD_DIR"/. /opt/frontend/; \
        elif [ -d "$FRONTEND_DIR/$FRONTEND_BUILD_DIR" ]; then \
            cp -r "$FRONTEND_DIR/$FRONTEND_BUILD_DIR"/. /opt/frontend/; \
        else \
            echo "Frontend build directory '$FRONTEND_BUILD_DIR' not found; skipping artifact copy."; \
        fi; \
    else \
        echo "No package.json found in $FRONTEND_DIR; skipping npm build."; \
        mkdir -p /opt/frontend; \
    fi && \
    # Drop node_modules to keep the stage lean before exporting artifacts
    rm -rf "$FRONTEND_DIR/node_modules"

##############################################
# Backend/runtime stage (Python + FastAPI)
##############################################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System packages required by pandas/reportlab/matplotlib and friends
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libjpeg62-turbo-dev \
        libfreetype6-dev \
        libpng-dev \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

# Copy application source
COPY . .

# Copy the built frontend assets from the Node stage (if any)
COPY --from=frontend-builder /opt/frontend ./src/static/dist

# Create a non-root user for better security
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.ui_server:app", "--host", "0.0.0.0", "--port", "8000"]
