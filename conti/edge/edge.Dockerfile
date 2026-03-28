FROM python:3.9-slim

WORKDIR /app

# Copy entire directories to maintain structure
COPY edge/ ./edge/
COPY shared/ ./shared/

# Install dependencies
RUN pip install --no-cache-dir -r edge/requirements.txt

# Set Python path to include app directory
ENV PYTHONPATH=/app

# Run the edge application
CMD ["python", "-m", "edge.edge_app"]
