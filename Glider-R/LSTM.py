import tensorflow as tf
from tensorflow.keras import layers

import numpy as np


def generate_opt_labels(memory_access_trace):
    """
    Generate labels for memory access trace using the OPT (Belady's) algorithm.
    Label 1 indicates 'cache-friendly' and 0 indicates 'cache-averse'.
    """
    labels = [0] * len(memory_access_trace)
    access_indices = {val: [] for val in set(memory_access_trace)}

    # Create a list of indices for each memory access
    for idx, access in enumerate(memory_access_trace):
        access_indices[access].append(idx)

    # Iterate through the memory access trace
    for idx, access in enumerate(memory_access_trace):
        future_accesses = access_indices[access]

        # Remove the current access from future_accesses
        future_accesses.pop(0)

        # If the current access will be accessed again in the future, label it as 'cache-friendly'
        if future_accesses:
            labels[idx] = 1

    return labels

def preprocess_data(memory_access_trace, sequence_length):
    # Split the trace into fixed-length overlapping sequences
    sequences = []
    half_sequence = sequence_length // 2
    for i in range(0, len(memory_access_trace) - sequence_length + 1, half_sequence):
        sequences.append(memory_access_trace[i:i + sequence_length])
    return np.array(sequences)

# Define the LSTM model with attention
class CacheReplacementModel(tf.keras.Model):
    def __init__(self, pc_vocab_size, embedding_dim, lstm_units):
        super(CacheReplacementModel, self).__init__()
        self.embedding = layers.Embedding(input_dim=pc_vocab_size, output_dim=embedding_dim)
        self.lstm = layers.LSTM(lstm_units, return_sequences=True)
        self.attention = layers.Attention(use_scale=True)
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.embedding(inputs)
        lstm_out = self.lstm(x)
        attention_out = self.attention([lstm_out, lstm_out])
        return self.output_layer(attention_out)



sequence_length = 2 * 10  # Update 'N' with the specific value from the paper

import os
print(os.getcwd())

memory_access_trace_path = "Glider-R/data/aster_163B.trace.xz"  # Your memory access trace data
f=open(memory_access_trace_path, 'rb')
memory_access_trace = np.load(f)


processed_data = preprocess_data(memory_access_trace, sequence_length)
pc_vocab_size = 1000  # Replace with the actual size from the paper
embedding_dim = 128  # Update if specified in the paper
lstm_units = 256     # Update if specified in the paper

model = CacheReplacementModel(pc_vocab_size, embedding_dim, lstm_units)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Assuming you have processed_data and labels
train_data = processed_data

train_labels = generate_opt_labels(memory_access_trace)

# Train the model
epochs = 10  # Number of epochs
batch_size = 32  # Batch size
model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)

# # Assuming you have test_data and test_labels
# test_data = [...]  # Test dataset
# test_labels = [...]  # Test labels

# # Evaluate the model
# model.evaluate(test_data, test_labels)
