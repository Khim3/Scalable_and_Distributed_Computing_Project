# Use a CPU-only PyTorch base image
FROM pytorch/pytorch

# Install Java, Spark, and necessary dependencies
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk wget python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment variable
ENV JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
ENV PATH="$JAVA_HOME/bin:$PATH"

# Install Spark 3.5.3
ENV SPARK_VERSION="3.5.3"
RUN wget https://archive.apache.org/dist/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-hadoop3.tgz && \
    tar -xzf spark-$SPARK_VERSION-bin-hadoop3.tgz -C /opt && \
    rm spark-$SPARK_VERSION-bin-hadoop3.tgz && \
    ln -s /opt/spark-$SPARK_VERSION-bin-hadoop3 /opt/spark

# Set Spark environment variables
ENV SPARK_HOME="/opt/spark"
ENV PATH="$SPARK_HOME/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Copy Python dependencies and install
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy the entire project into the container
COPY . /app

# Expose Streamlit's default port
EXPOSE 8501

# Command to run your Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
