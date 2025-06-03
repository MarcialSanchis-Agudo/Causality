import tensorflow as tf
import math

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = tf.keras.layers.Dense(d_model)
        self.W_k = tf.keras.layers.Dense(d_model)
        self.W_v = tf.keras.layers.Dense(d_model)
        self.W_o = tf.keras.layers.Dense(d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask):
        matmul_qk = tf.matmul(Q, K, transpose_b=True)
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, V)
        return output

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[2]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
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
        return self.W_o(concat_attention)


class PositionWiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, activation="relu"):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = tf.keras.layers.Dense(d_ff)
        self.fc2 = tf.keras.layers.Dense(d_model)

        if activation == "relu":
            self.act = tf.keras.layers.ReLU()
        elif activation == "gelu":
            self.act = tf.keras.layers.Activation(tf.nn.gelu)
        elif activation == "elu":
            self.act = tf.keras.layers.ELU()
        else:
            raise ValueError("Unsupported activation function")

    def call(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, activation)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        attn_output = self.mha(Q=x, K=x, V=x, mask=mask)
        out1 = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output, training=training))
        return out2


class CrossSpaceTimeEasyAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, seq_len, num_heads):
        super(CrossSpaceTimeEasyAttention, self).__init__()
        assert d_model % num_heads == 0 and seq_len % num_heads == 0, "d_model and seq_len must be divisible by num_heads"

        self.d_model = d_model
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_t = seq_len // num_heads

        self.alpha = self.add_weight(shape=(num_heads, seq_len, seq_len), initializer='glorot_uniform', trainable=True)
        self.wvl = self.add_weight(shape=(seq_len, seq_len), initializer='glorot_uniform', trainable=True)
        self.wvr = self.add_weight(shape=(num_heads, d_model, d_model), initializer='glorot_uniform', trainable=True)

        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    def call(self, x):
        B = tf.shape(x)[0]

        x_attn = self.mha(Q=x, K=x, V=x)

        x_wvl = tf.matmul(tf.tile(tf.expand_dims(self.wvl, 0), [B, 1, 1]), x)

        x_time = tf.reshape(x_wvl, [B, self.num_heads, self.d_t, self.d_model])
        wvr_tiled = tf.tile(tf.expand_dims(self.wvr, 0), [B, 1, 1, 1])
        x_time = tf.einsum('bhtd,bhdf->bhtf', x_time, wvr_tiled)
        x_time = tf.reshape(x_time, [B, self.seq_len, self.d_model])

        x_space = tf.reshape(x_time, [B, self.seq_len, self.num_heads, self.d_k])
        x_space = tf.transpose(x_space, perm=[0, 2, 1, 3])
        alpha_tiled = tf.tile(tf.expand_dims(self.alpha, 0), [B, 1, 1, 1])
        x_space = tf.matmul(alpha_tiled, x_space)
        x_space = tf.transpose(x_space, perm=[0, 2, 1, 3])
        x_space = tf.reshape(x_space, [B, self.seq_len, self.d_model])

        return x_space + x_attn


class EasyEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, attn_type, d_model, seq_len, num_heads, d_ff, dropout_rate=0.1, activation="relu"):
        super(EasyEncoderLayer, self).__init__()
        if attn_type == 'easy':
            self.attn = EasyAttention(d_model, seq_len, num_heads)
        elif attn_type == 'cross':
            self.attn = CrossSpaceTimeEasyAttention(d_model, seq_len, num_heads)
        else:
            raise ValueError("Unsupported attention type")

        self.ffn = PositionWiseFeedForward(d_model, d_ff, activation)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        attn_output = self.attn(x)
        out1 = self.norm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        out2 = self.norm2(out1 + self.dropout2(ffn_output, training=training))
        return out2


class EasyAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, seq_len, num_heads):
        super(EasyAttention, self).__init__()
        assert d_model % num_heads == 0 and seq_len % num_heads == 0, "d_model and seq_len must be divisible by num_heads"

        self.d_model = d_model
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_t = seq_len // num_heads

        self.alpha = self.add_weight(shape=(num_heads, seq_len, seq_len), initializer='glorot_uniform', trainable=True)
        self.wvl = self.add_weight(shape=(seq_len, seq_len), initializer='glorot_uniform', trainable=True)
        self.wvr = self.add_weight(shape=(num_heads, d_model, d_model), initializer='glorot_uniform', trainable=True)

        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    def call(self, x):
        B = tf.shape(x)[0]
        x_attn = self.mha(Q=x, K=x, V=x)

        x_wvl = tf.matmul(tf.tile(tf.expand_dims(self.wvl, 0), [B, 1, 1]), x)

        x_time = tf.reshape(x_wvl, [B, self.num_heads, self.d_t, self.d_model])
        wvr_tiled = tf.tile(tf.expand_dims(self.wvr, 0), [B, 1, 1, 1])
        x_time = tf.einsum('bhid,bhde->bhie', x_time, wvr_tiled)
        x_time = tf.reshape(x_time, [B, self.seq_len, self.d_model])

        x_space = tf.reshape(x_time, [B, self.seq_len, self.num_heads, self.d_k])
        x_space = tf.transpose(x_space, perm=[0, 2, 1, 3])

        alpha_tiled = tf.tile(tf.expand_dims(self.alpha, 0), [B, 1, 1, 1])
        x_space = tf.matmul(alpha_tiled, x_space)
        x_space = tf.transpose(x_space, perm=[0, 2, 1, 3])
        x_space = tf.reshape(x_space, [B, self.seq_len, self.d_model])

        return x_space + x_attn
