# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Sandbox Agent Docker Image
# Runs as root to allow apt-get install at runtime.
# Contains 3 HTTP services: Jupyter Kernel Gateway (8888),
# Browser Server (8889), Shell Server (8890).

FROM python:3.12-slim

# Metadata
LABEL maintainer="NVIDIA Corporation"
LABEL description="Sandbox Agent execution environment with persistent IPython, browser, and shell services"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Basic utilities
    curl \
    wget \
    git \
    jq \
    unzip \
    ca-certificates \
    # Compilation dependencies (pip install C extensions)
    build-essential \
    # GAIA high-frequency
    ffmpeg \
    poppler-utils \
    # Playwright browser dependencies
    libnss3 \
    libnspr4 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    # Fonts (browser rendering)
    fonts-liberation \
    fonts-dejavu-core \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    # Architecture: IPython kernel
    ipykernel \
    jupyter_kernel_gateway \
    # Architecture: HTTP services
    fastapi \
    uvicorn \
    # GAIA core - data processing
    pandas \
    numpy \
    matplotlib \
    pillow \
    # GAIA core - network
    requests \
    httpx \
    beautifulsoup4 \
    # GAIA core - file formats
    openpyxl \
    pyyaml \
    python-dateutil \
    # Browser automation
    playwright \
    # Web search (agent calls via IPython)
    tavily-python

# Install Playwright browser (Chromium only)
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers
RUN mkdir -p ${PLAYWRIGHT_BROWSERS_PATH} && \
    playwright install chromium --with-deps

# Create workspace directories
RUN mkdir -p /workspace/input \
             /workspace/output \
             /workspace/temp \
             /workspace/downloads

# Set working directory
WORKDIR /workspace

# Expose service ports
EXPOSE 8888 8889 8890

# Default command - keep container running
CMD ["/bin/bash"]
