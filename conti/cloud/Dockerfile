FROM python:3.9-slim

WORKDIR /app

# Copy entire directories to maintain structure
COPY cloud/ ./cloud/
COPY shared/ ./shared/

# Install dependencies
RUN pip install --no-cache-dir -r cloud/requirements.txt

# Set Python path to include app directory
ENV PYTHONPATH=/app

# Run the cloud application
CMD ["python", "-m", "cloud.cloud_app"]
