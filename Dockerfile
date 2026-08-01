FROM python:3.11-slim

# System libs cartopy/shapely need (GEOS, PROJ) + build tools for any
# packages without prebuilt wheels for this platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgeos-dev \
    libproj-dev \
    proj-bin \
    proj-data \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets $PORT at runtime; app.py reads it.
CMD ["python", "app.py"]
