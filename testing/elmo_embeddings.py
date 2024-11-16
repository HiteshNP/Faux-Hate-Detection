import tensorflow as tf
import tensorflow_hub as hub

# Load the Elmo model
elmo = hub.load("https://tfhub.dev/google/elmo/3")

documents = ["I will show you a valid point of reference and talk to the point",
             "Where have you placed the point"]

# Generate Elmo embeddings
embeddings = elmo(documents)

# To get the embeddings of the first word "point" in the first sentence
print("Word embeddings for the first 'point' in the first sentence")
print(embeddings[0][6])  # Index 6 corresponds to the word "point"
