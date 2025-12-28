import tensorflow as tf

try: from transformer import TransformerBlock, PositionalEncoding
except Exception as e: print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

########################################################################################

@tf.keras.saving.register_keras_serializable(package="MyLayers")
class RNNDecoder(tf.keras.layers.Layer):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO:
        # Now we will define image and word embedding, decoder, and classification layers
        self.word_embedding = tf.keras.layers.Embedding(self.vocab_size, self.hidden_size, embeddings_initializer='he_normal')
        self.decoder_layer = tf.keras.layers.GRU(self.hidden_size, return_sequences=True)
        self.image_to_hidden = tf.keras.layers.Dense(self.hidden_size)
        self.classification = tf.keras.layers.Dense(self.vocab_size)
        
    def call(self, encoded_images, captions):
        """
        :param encoded_images: tensor of shape [BATCH_SIZE x 2048]
        :param captions: tensor of shape [BATCH_SIZE x WINDOW_SIZE]
        :return: batch logits of shape [BATCH_SIZE x WINDOW_SIZE x VOCAB_SIZE]
        """

        # TODO:
        x = self.word_embedding(captions)
        initial_state = self.image_to_hidden(encoded_images)
        x = self.decoder_layer(x, initial_state=initial_state)

        logits = self.classification(x)
        return logits


    def get_config(self):
        base_config = super().get_config()
        config = {k:getattr(self, k) for k in ["vocab_size", "hidden_size", "window_size"]}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

########################################################################################

@tf.keras.saving.register_keras_serializable(package="MyLayers")
class TransformerDecoder(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO:
        # Now we will define image and word embedding, positional encoding, tramnsformer decoder, and classification layers
        self.image_embedding = tf.keras.layers.Dense(self.hidden_size)
        self.word_embedding = PositionalEncoding(self.vocab_size, self.hidden_size, self.window_size)
        self.transformer_decoder = TransformerBlock(self.hidden_size,multiheaded=True)
        self.classification = tf.keras.layers.Dense(self.vocab_size)


    def call(self, encoded_images, captions):
        """
        :param encoded_images: tensor of shape [BATCH_SIZE x 2048]
        :param captions: tensor of shape [BATCH_SIZE x WINDOW_SIZE]
        :return: batch logits of shape [BATCH_SIZE x WINDOW_SIZE x VOCAB_SIZE]
        """
        # TODO
        word_embedding = self.word_embedding(captions)
        image_embedding = self.image_embedding(encoded_images)
        image_embedding = tf.expand_dims(image_embedding, axis=1)
        image_embedding = tf.tile(image_embedding, [1, self.window_size, 1])
        x = self.transformer_decoder(word_embedding, image_embedding)

        logits = self.classification(x)
        return logits

    def get_config(self):
        base_config = super().get_config()
        config = {k:getattr(self, k) for k in ["vocab_size", "hidden_size", "window_size"]}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)    
