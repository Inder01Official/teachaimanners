import tensorflow as tf
from keras.preprocessing.text import Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
import numpy as np

# Sample book text
book_text = """
'''Mike: Hi Sarah, long time no see. How are you?
    Sarah: Hi Mike, I'm great. You?
    Mike: It's a beautiful day; couldn't be better.
    Sarah: Yeah, the weather is really nice today. Have you been up to anything interesting lately?
    Mike: Actually, I went on a hiking trip last weekend. It was amazing. I highly recommend it.
    Sarah: That sounds like a lot of fun. Where did you go?
    Mike: We went to the mountains up north. The scenery was breathtaking.
    Sarah: I'll have to plan a trip up there sometime. Thanks for the recommendation!
    Mike: No problem. So, how about you? What have you been up to?
    Sarah: Not too much, just been busy with work. But I'm hoping to take a vacation soon.
    Mike: That sounds like a good idea. Any ideas on where you want to go?
    Sarah: I'm thinking about going to the beach. I could use some time in the sun and sand.
    Mike: Sounds like a great plan. I hope you have a good time.
    Sarah: Thanks, Mike. It was good seeing you.
    Mike: Same here, Sarah. Take care!
    ''',
        '''Jack: Hey Emily! How's it going?
    Emily: Hey Jack! It's going pretty well. How about you?
    Jack: Not bad, thanks for asking. So, what have you been up to lately?
    Emily: Well, I recently started taking a painting class, which has been really fun. What about you?
    Jack: That's cool. I've been trying to get into running. It's been tough, but I'm starting to enjoy it.
"""

# Preprocess the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([book_text])
sequences = tokenizer.texts_to_sequences([book_text])
vocab_size = len(tokenizer.word_index) + 1

# Create input-output pairs
input_sequences = []
for sequence in sequences:
    for i in range(1, len(sequence)):
        n_gram_sequence = sequence[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Split into input and output
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# Convert output to one-hot encoding
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 100, input_length=max_sequence_len-1))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)))
model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=50, verbose=1)

# Save the trained model
model.save("trained_model.h5")

# Load the saved model
loaded_model = tf.keras.models.load_model("trained_model.h5")

# Generate text
seed_text = "This is"
next_words = 10

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = loaded_model.predict(token_list, verbose=0)
    predicted_word_index = tf.argmax(predicted, axis=-1).numpy()[0]
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            output_word = word
            break
    seed_text += " " + output_word

print(seed_text)
