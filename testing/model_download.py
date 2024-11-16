import kagglehub

# Download latest version
path = kagglehub.model_download("google/elmo/tensorFlow1/elmo")

print("Path to model files:", path)