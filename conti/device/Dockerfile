FROM python:3.9-slim

WORKDIR /app

# Copy entire directories to maintain structure
COPY device/ ./device/
COPY shared/ ./shared/

# Install dependencies
RUN pip install --no-cache-dir -r device/requirements.txt

# Set Python path to include app directory
ENV PYTHONPATH=/app

# Run the device application
CMD ["python", "-m", "device.device_app"]
