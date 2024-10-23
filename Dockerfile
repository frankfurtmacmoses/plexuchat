# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /welcome

# Copy the current directory contents into the container at /app
COPY . /welcome.py

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV OPENAI_API_KEY=

# Run streamlit
CMD ["streamlit", "run", "welcome.py", "--server.port=8501"]
