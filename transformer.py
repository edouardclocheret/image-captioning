import math
import numpy as np
import tensorflow as tf

@tf.keras.saving.register_keras_serializable(package="transformer_layers")
class AttentionMatrix(tf.keras.layers.Layer):

    def __init__(self, *args, use_mask=False, **kwargs):
        super().__init__(*args, **kwargs)
        # Mask is [batch_size x window_size_queries x window_size_keys]

        self.use_mask = use_mask

    def call(self, inputs):
        """
        STUDENT MUST WRITE:

        This functions runs a single attention head.

        :param K: is [batch_size x window_size_keys x embedding_size]
        :param Q: is [batch_size x window_size_queries x embedding_size]
        :return: attention matrix
        """
        K, Q = inputs
        window_size_queries = Q.get_shape()[1]  # window size of queries
        window_size_keys    = K.get_shape()[1]  # window size of keys
        embedding_size_keys = K.get_shape()[2]

        mask = tf.convert_to_tensor(
            value=np.transpose(np.tril(np.ones((window_size_queries, window_size_keys)) * np.NINF, -1), (1, 0)),
            dtype=tf.float32)
        atten_mask = tf.tile(tf.reshape(mask, [-1, window_size_queries, window_size_keys]), [tf.shape(input=K)[0], 1, 1])

        # TODO:
        # 1) compute self-attention weights (if use_mask==True, make sure to add the attention mask before softmax)
        # 2) return the attention matrix

        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(embedding_size_keys, tf.float32))
        if self.use_mask :
            return tf.nn.softmax(scores + atten_mask, axis=-1)
        else:
            return tf.nn.softmax(scores, axis=-1)

@tf.keras.saving.register_keras_serializable(package="transformer_layers")
class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, is_self_attention, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)
        self.use_mask = is_self_attention

        # TODO:
        # Initialize the attention matrices (what size tensor should they produce?)
        self.WQ = self.add_weight(name ="WQ", shape=(input_size, output_size), initializer="glorot_uniform", trainable=True)
        self.WK = self.add_weight(name = "WK", shape=(input_size, output_size), initializer="glorot_uniform", trainable=True)
        self.WV = self.add_weight(name = "WV", shape=(input_size, output_size), initializer="glorot_uniform", trainable=True)
        
        self.attention_matrix = AttentionMatrix(use_mask=self.use_mask)
        
        
    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        This functions runs a single attention head.

        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """

        # TODO
        K = inputs_for_keys @ self.WK
        Q = inputs_for_queries @ self.WQ
        V = inputs_for_values @ self.WV
        attention_weights = self.attention_matrix((K, Q))
        return attention_weights @ V

@tf.keras.saving.register_keras_serializable(package="transformer_layers")
class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, emb_sz, use_mask, **kwargs):
        super(MultiHeadedAttention, self).__init__(**kwargs)

        # Initialize Attention Heads Here
        self.head_1 = AttentionHead(emb_sz, emb_sz//3 , use_mask)
        self.head_2 = AttentionHead(emb_sz, emb_sz//3, use_mask)
        self.head_3 = AttentionHead(emb_sz, emb_sz//3, use_mask)
        self.output_linear = tf.keras.layers.Dense(emb_sz)


    # @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        This functions runs a multiheaded attention layer.

        Requirements:
            - 3 different heads of size embed_sz/3

        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """
        z1 = self.head_1(inputs_for_keys, inputs_for_values, inputs_for_queries)
        z2 = self.head_2(inputs_for_keys, inputs_for_values, inputs_for_queries)
        z3 = self.head_3(inputs_for_keys, inputs_for_values, inputs_for_queries)
        concat = tf.concat([z1, z2, z3], axis=-1)
        return self.output_linear(concat)

@tf.keras.saving.register_keras_serializable(package="transformer_layers")
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, emb_sz, multiheaded=False, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        # TODO
        # Use multiheaded attention if multiheaded is True!
        if multiheaded :
            self.self_attention = MultiHeadedAttention(emb_sz, use_mask=True) # Masked self attention 
            self.cross_attention = MultiHeadedAttention(emb_sz, use_mask=False) # Cross is not masked
        else:
            self.self_attention = AttentionHead(emb_sz, emb_sz, is_self_attention=True)
            self.cross_attention = AttentionHead(emb_sz, emb_sz, is_self_attention=False)
        
        self.normalization_1 = tf.keras.layers.LayerNormalization()
        self.normalization_2 = tf.keras.layers.LayerNormalization()

        #feedforward
        self.dense_1 = tf.keras.layers.Dense(emb_sz*4) #in the paper they have a ratio of 4
        self.relu = tf.keras.layers.ReLU()
        self.dense_2 = tf.keras.layers.Dense(emb_sz)

        self.normalization_3 = tf.keras.layers.LayerNormalization()


    @tf.function
    def call(self, inputs, context_sequence):
        """
        This functions calls a transformer block.

        :param inputs: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :param context_sequence: tensor of shape [BATCH_SIZE x CONTEXT_SEQ_LENGTH x EMBEDDING_SIZE ]
        :return: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        """

        # TODO

        #self attention
        masked_attention = self.self_attention(inputs, inputs, inputs)
        add_norm_1 = self.normalization_1(inputs + masked_attention)

        #cross attention
        cross_attention  = self.cross_attention(context_sequence, context_sequence, add_norm_1)
        add_norm_2 = self.normalization_2(add_norm_1 + cross_attention)

        #feedforward
        x = self.dense_1(add_norm_2)
        x = self.relu(x)
        x = self.dense_2(x)
        add_norm_3 = self.normalization_3(add_norm_2 + x)
        
        return add_norm_3


#Inspired by https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer

@tf.keras.saving.register_keras_serializable(package="transformer_layers", name="positional_encoding")
def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)


@tf.keras.saving.register_keras_serializable(package="transformer_layers")
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)
        self.pos_encoding = positional_encoding(window_size, embed_size)[..., :window_size, :]

    def call(self, x):
        x_embed = self.embedding(x)
        x_embed *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
        x_embed += self.pos_encoding
        return x_embed