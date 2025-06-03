import tensorflow as tf
import math

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Dense layers for Q, K, V
        self.W_q = tf.keras.layers.Dense(d_model)
        self.W_k = tf.keras.layers.Dense(d_model)
        self.W_v = tf.keras.layers.Dense(d_model)
        self.W_o = tf.keras.layers.Dense(d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask):
        matmul_qk = tf.matmul(Q, K, transpose_b=True)
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  # Add large negative value to masked positions

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, V)
        return output

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # [B, H, S, D]

    def combine_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[2]
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # [B, S, H, D]
        return tf.reshape(x, (batch_size, seq_len, self.d_model))

    def call(self, Q, K, V, mask=None):
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention = self.scaled_dot_product_attention(Q, K, V, mask)
        concat_attention = self.combine_heads(attention)

        output = self.W_o(concat_attention)
        return output
