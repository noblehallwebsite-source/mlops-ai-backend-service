FROM python:3.11-slim

WORKDIR /app

# 1. Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# 2. Force the installation of the lightweight CPU version of torch FIRST
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install the rest of the dependencies. 
# Pip will see CPU-torch is already installed and will skip the heavy CUDA binaries.
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your application code
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]