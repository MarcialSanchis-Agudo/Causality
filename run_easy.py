#!/usr/bin/env python3
# https://github.com/tensorflow/agents/blob/master/tf_agents/examples/ppo/schulman17/train_eval_lib.py
# https://github.com/tensorflow/agents/blob/master/tf_agents/agents/ppo/examples/v2/train_eval_clip_agent.py

# pylint: disable=protected-access
import os
import random
import time
import contextlib
import numpy as np
import tensorflow as tf

from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.train.utils import spec_utils, strategy_utils
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.networks import value_network
from tf_agents.eval import metric_utils
from tf_agents.policies import policy_saver
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.distributions.distribution_projection_network import DistributionProjectionNetwork

from smartsim.log import get_logger
import absl.logging

from params import params, env_params
from smartsod2d.init_smartsim import init_smartsim
from sod_env_cyl import SodEnvCyl
from smartsod2d.history import History
from smartsod2d.utils import print_params, bcolors, params_str, numpy_str, deactivate_tf_gpus

# --------------------------- Attention / Transformer Components ---------------------------

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = tf.keras.layers.Dense(d_model, name="W_q")
        self.W_k = tf.keras.layers.Dense(d_model, name="W_k")
        self.W_v = tf.keras.layers.Dense(d_model, name="W_v")
        self.W_o = tf.keras.layers.Dense(d_model, name="W_o")

    def split_heads(self, x):
        # x: (batch, seq_len, d_model)
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch, num_heads, seq_len, d_k)

    def combine_heads(self, x):
        # x: (batch, num_heads, seq_len, d_k)
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[2]
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # (batch, seq_len, num_heads, d_k)
        return tf.reshape(x, (batch_size, seq_len, self.d_model))  # (batch, seq_len, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask):
        # Q, K, V: (batch, num_heads, seq_len, d_k)
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scores = tf.matmul(Q, K, transpose_b=True)  # (batch, num_heads, seq_len, seq_len)
        scores = scores / tf.math.sqrt(dk)
        if mask is not None:
            scores += (mask * -1e9)
        weights = tf.nn.softmax(scores, axis=-1)  # (batch, num_heads, seq_len, seq_len)
        output = tf.matmul(weights, V)             # (batch, num_heads, seq_len, d_k)
        return output

    def call(self, Q, K, V, mask=None):
        # Q, K, V: (batch, seq_len, d_model)
        Q_proj = self.W_q(Q)  # (batch, seq_len, d_model)
        K_proj = self.W_k(K)
        V_proj = self.W_v(V)

        Q_split = self.split_heads(Q_proj)  # (batch, num_heads, seq_len, d_k)
        K_split = self.split_heads(K_proj)
        V_split = self.split_heads(V_proj)

        attn = self.scaled_dot_product_attention(Q_split, K_split, V_split, mask)
        # → (batch, num_heads, seq_len, d_k)
        concat = self.combine_heads(attn)     # (batch, seq_len, d_model)
        return self.W_o(concat)              # (batch, seq_len, d_model)


class CrossSpaceTimeEasyAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, seq_len, num_heads, name="cross_space_time_easy_attn"):
        super(CrossSpaceTimeEasyAttention, self).__init__(name=name)
        assert d_model % num_heads == 0 and seq_len % num_heads == 0, "d_model and seq_len must be divisible by num_heads"

        self.d_model = d_model
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.d_k = d_model // num_heads    # feature split per head
        self.d_t = seq_len // num_heads    # time‐chunk per head

        # α ∈ ℝ^{H×T×T}: per‐head mixing across time (no softmax)
        self.alpha = self.add_weight(
            shape=(num_heads, seq_len, seq_len),
            initializer="glorot_uniform",
            trainable=True,
            name="alpha"
        )
        # W_lv ∈ ℝ^{T×T}: linear mixing along time axis
        self.wvl = self.add_weight(
            shape=(seq_len, seq_len),
            initializer="glorot_uniform",
            trainable=True,
            name="wvl"
        )
        # W_vr ∈ ℝ^{H×D×D}: per‐head feature mixing
        self.wvr = self.add_weight(
            shape=(num_heads, d_model, d_model),
            initializer="glorot_uniform",
            trainable=True,
            name="wvr"
        )

        # A standard multi‐head self‐attention in parallel
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, name="csta_mha")

    def call(self, x):
        # x: (batch, seq_len, d_model)
        B = tf.shape(x)[0]

        # (1) Standard self‐attention branch
        x_attn = self.mha(Q=x, K=x, V=x)  # → (B, T, D)

        # (2) Time‐mix branch: x_wvl = W_lv ⋅ x along time axis
        wvl_tiled = tf.tile(tf.expand_dims(self.wvl, 0), [B, 1, 1])  # (B, T, T)
        x_wvl = tf.matmul(wvl_tiled, x)                               # (B, T, D)

        # (2a) Reshape into (B, H, d_t, D)
        x_time = tf.reshape(x_wvl, [B, self.num_heads, self.d_t, self.d_model])

        # (2b) Per‐head feature mixing: W_vr_h ∈ ℝ^{D×D}, broadcast to (B, H, D, D)
        wvr_tiled = tf.tile(tf.expand_dims(self.wvr, 0), [B, 1, 1, 1])  # (B, H, D, D)
        # x_time: (B, H, d_t, D); apply per‐head matmul:
        x_time = tf.einsum('bhtd,bhdf->bhtf', x_time, wvr_tiled)        # (B, H, d_t, D)
        x_time = tf.reshape(x_time, [B, self.seq_len, self.d_model])   # (B, T, D)

        # (3) Space‐mix branch: split features into heads, then mix across time with α
        x_space = tf.reshape(x_time, [B, self.seq_len, self.num_heads, self.d_k])
        x_space = tf.transpose(x_space, perm=[0, 2, 1, 3])  # (B, H, T, d_k)

        # α_tiled: (B, H, T, T)
        alpha_tiled = tf.tile(tf.expand_dims(self.alpha, 0), [B, 1, 1, 1])
        x_space = tf.matmul(alpha_tiled, x_space)           # (B, H, T, d_k)
        x_space = tf.transpose(x_space, perm=[0, 2, 1, 3])   # (B, T, H, d_k)
        x_space = tf.reshape(x_space, [B, self.seq_len, self.d_model])  # (B, T, D)

        # (4) Sum the easy‐attention output (x_space) with standard self‐attention (x_attn)
        return x_space + x_attn  # (B, T, D)


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Standard sinusoidal positional encoding.  Adds a fixed sin/cos encoding
    of shape (1, seq_len, d_model) to any input x of shape (batch, seq_len, d_model).
    """
    def __init__(self, seq_len, d_model, name="positional_encoding"):
        super(PositionalEncoding, self).__init__(name=name)
        # Precompute the T×D matrix once
        pos = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)     # (T, 1)
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]          # (1, D)
        angle_rates = 1 / tf.pow(10000.0, (2 * tf.floor(i / 2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates                                  # (T, D)

        # apply sin to even indices, cos to odd indices
        sines = tf.sin(angle_rads[:, 0::2])
        coses = tf.cos(angle_rads[:, 1::2])

        pe = tf.concat([sines, coses], axis=-1)                          # (T, D)
        self.pe = pe[tf.newaxis, ...]  # (1, T, D) to broadcast over batch

    def call(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]


class AttentionActorNetwork(network.Network):
    """
    Custom actor network with improved embedding:
      - Accepts observations as either (batch, S) or (batch, T, S),
        where S = number of sensors, T = time‐window length.
      - Embeds spatially via a Dense: (S → d_model).
      - Adds a sinusoidal positional encoding over time.
      - Applies one or more CrossSpaceTimeEasyAttention layers.
      - Flattens → MLP trunk (two Dense layers) → DistributionProjectionNetwork.
    """
    def __init__(
        self,
        input_tensor_spec,
        action_spec,
        seq_len: int,
        d_model: int = 128,
        num_heads: int = 8,
        d_ff: int = 512,
        num_cross_attn_layers: int = 1,
        activation: str = "relu",
        name: str = "AttentionActorNetwork",
    ):
        """
        Args:
          input_tensor_spec: A TensorSpec describing the shape/dtype of observations.
                             Should be either (S,) for a single time‐step,
                             or (T, S) for a temporal window of sensors.
          action_spec:       A BoundedTensorSpec for actions.
          seq_len:           The “time‐window” length T that we expect.
          d_model:           Hidden dimension for all attention heads/features.
          num_heads:         Number of attention heads (must divide d_model and seq_len).
          d_ff:              Hidden size of the trunk MLP.
          num_cross_attn_layers: How many CrossSpaceTimeEasyAttention layers to stack.
          activation:        Activation to use in the trunk (“relu”/“gelu”/“elu”).
        """
        super(AttentionActorNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name
        )
        self._action_spec = action_spec
        self._seq_len = seq_len
        self._d_model = d_model
        self._num_heads = num_heads
        self._num_cross_attn_layers = num_cross_attn_layers

        # Determine sensor‐dimension S from input_tensor_spec:
        obs_shape = input_tensor_spec.shape  # e.g. (S,) or (T, S)
        if len(obs_shape) == 2:
            T_spec, S_spec = obs_shape
            if T_spec is not None and S_spec is not None:
                assert int(T_spec) == seq_len, f"spec time‐length {T_spec} ≠ seq_len {seq_len}"
            self._num_sensors = int(S_spec)
        elif len(obs_shape) == 1:
            # Only one time‐step provided; we treat T=1
            self._num_sensors = int(obs_shape[0])
        else:
            raise ValueError("input_tensor_spec must have shape (S,) or (T, S).")

        # 1) Spatial projection: map each of the S sensors → d_model features
        #    This layer will be applied to each time‐step if we have T>1.
        self._spatial_proj = tf.keras.layers.Dense(
            units=d_model,
            activation=None,
            name="spatial_proj",
            kernel_initializer=tf.keras.initializers.Orthogonal()
        )

        # 2) Sinusoidal positional encoding over time (length = seq_len)
        self._pos_enc = PositionalEncoding(seq_len=seq_len, d_model=d_model, name="pos_enc")

        # 3) Stack of CrossSpaceTimeEasyAttention layers
        self._csta_layers = []
        for i in range(num_cross_attn_layers):
            self._csta_layers.append(
                CrossSpaceTimeEasyAttention(
                    d_model=d_model,
                    seq_len=seq_len,
                    num_heads=num_heads,
                    name=f"cross_space_time_attn_{i}"
                )
            )

        # 4) Flatten to (batch, T * d_model)
        self._flatten = tf.keras.layers.Reshape((seq_len * d_model,), name="flatten_after_csta")

        # 5) MLP trunk: two Dense layers → (batch, d_ff)
        self._trunk_dense_1 = tf.keras.layers.Dense(
            units=d_ff,
            activation=activation,
            name="trunk_dense_1",
            kernel_initializer=tf.keras.initializers.Orthogonal(),
        )
        self._trunk_dense_2 = tf.keras.layers.Dense(
            units=d_ff,
            activation=activation,
            name="trunk_dense_2",
            kernel_initializer=tf.keras.initializers.Orthogonal(),
        )

        # 6) DistributionProjectionNetwork for continuous actions
        action_dim = tensor_spec.from_spec(action_spec).shape[0]
        self._dist_proj = DistributionProjectionNetwork(
            sample_spec=action_spec,
            convert_to_dtype=tf.float32,
            loc_layer_params=None,            # mean is linear
            scale_layer_params=[action_dim],  # log_std per action‐dim
            scale_distribution_fn=lambda x: tf.exp(x),
            name="dist_proj"
        )

    def call(self, observations, step_type=None, network_state=()):
        """
        Args:
          observations: A Tensor of shape
            - (batch, S)        if T=1, or
            - (batch, T, S)     if T>1 (temporal window of sensor readings).
          step_type:    Ignored here (for TF‐Agents compatibility).
          network_state: Ignored (no RNN state).

        Returns:
          action_dist:   A tfp.Distribution (e.g. Gaussian) with shape (batch, action_dim).
          network_state: (), since we have no recurrent state.
        """
        x = tf.cast(observations, tf.float32)
        # If shape is (batch, S), expand to (batch, 1, S)
        if tf.rank(x) == 2:
            x = tf.expand_dims(x, axis=1)  # → (batch, 1, S)

        # Now x: (batch, T, S), where T = self._seq_len
        # 1) Spatial projection applied to each of the T time‐steps:
        #    Dense on last axis S→d_model automatically broadcasts over time.
        #    Result: (batch, T, d_model)
        x_spat = self._spatial_proj(x)

        # 2) Add positional encoding over time:
        #    PositionalEncoding expects (batch, T, d_model)
        x_pe = self._pos_enc(x_spat)  # → (batch, T, d_model)

        # 3) Pass through N CrossSpaceTimeEasyAttention layers:
        x_csta = x_pe
        for csta in self._csta_layers:
            x_csta = csta(x_csta)  # (batch, T, d_model)

        # 4) Flatten to (batch, T * d_model)
        x_flat2 = self._flatten(x_csta)  # → (batch, T*d_model)

        # 5) MLP trunk → (batch, d_ff)
        h1 = self._trunk_dense_1(x_flat2)  # (batch, d_ff)
        h2 = self._trunk_dense_2(h1)       # (batch, d_ff)

        # 6) DistributionProjectionNetwork → a tfp.Distribution
        action_dist = self._dist_proj(h2)  # e.g. Normal(batch, action_dim)
        return action_dist, network_state

# --------------------------- Utils ---------------------------
absl.logging.set_verbosity(absl.logging.ERROR)
logger = get_logger(__name__)
cwd = os.path.dirname(os.path.realpath(__file__))
deactivate_tf_gpus()  # deactivate TF for GPUs
if params["use_XLA"]:  # activate XLA for performance
    os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
    os.environ['XLA_FLAGS'] = "--xla_hlo_profile"
    tf.config.optimizer.set_jit(True)
    tf.function(jit_compile=True)

# Write summary to TensorBoard
train_dir = os.path.join(cwd, "train")
summary_writer = tf.summary.create_file_writer(train_dir, flush_millis=1000)
summary_writer.set_as_default()

# Print simulation params
print_params(params)

# --------------------------- RL setup ---------------------------
# Init SmartSim framework: Experiment and Orchestrator (database)
exp, hosts, db, db_is_clustered = init_smartsim(
    port=params["port"],
    network_interface=params["network_interface"],
    launcher=params["launcher"],
    run_command=params["run_command"],
)

# Init environment
collect_py_env = SodEnvCyl(
    exp,
    db,
    hosts,
    "sod2d",
    cwd,
    cfd_n_envs=params["cfd_n_envs"],
    mode="collect",
    **env_params,
)
collect_env = tf_py_environment.TFPyEnvironment(collect_py_env)

global_step = tf.compat.v1.train.get_or_create_global_step()
observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
    spec_utils.get_tensor_specs(collect_env)
)

logger.info(f'Observation Spec:\n{observation_tensor_spec}')
logger.info(f'Action Spec:\n{action_tensor_spec}')
logger.info(f'Time Spec:\n{time_step_tensor_spec}')

# ---------------------- Replace sequential actor_net with AttentionActorNetwork ----------------------

# Instead of:
# actor_net_builder = ppo_actor_network.PPOActorNetwork()
# actor_net = actor_net_builder.create_sequential_actor_net(params["net"], action_tensor_spec)

# We instantiate our custom attention actor with improved embedding:
seq_len_from_env = params["seq_len"]                       # e.g. how many past time‐steps we include
d_model        = params.get("d_model", 128)
num_heads      = params.get("num_heads", 8)
d_ff           = params.get("d_ff", 512)
num_cross_attn_layers = params.get("num_attn_layers", 1)   # baseline: 1 cross-attn layer

actor_net = AttentionActorNetwork(
    input_tensor_spec=observation_tensor_spec,
    action_spec=action_tensor_spec,
    seq_len=seq_len_from_env,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_cross_attn_layers=num_cross_attn_layers,
    activation="relu",
)

value_net = value_network.ValueNetwork(
    observation_tensor_spec,
    fc_layer_params=params["net"],
    kernel_initializer=tf.keras.initializers.Orthogonal()
)

# For distribution strategy, networks and agent have to be initialized within strategy.scope
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params["learning_rate"])
strategy = strategy_utils.get_strategy(tpu=False, use_gpu=False)
if strategy:
    context = strategy.scope()
else:
    context = contextlib.nullcontext()  # placeholder

with context:
    # Set TF random seed within strategy to obtain reproducible results
    random.seed(params["seed"])
    np.random.seed(params["seed"])
    tf.random.set_seed(params["seed"])

    # PPO Agent with our attention‐based actor_net
    agent = ppo_clip_agent.PPOClipAgent(
        time_step_tensor_spec,
        action_tensor_spec,
        optimizer=optimizer,
        actor_net=actor_net,
        value_net=value_net,
        entropy_regularization=0.0,
        importance_ratio_clipping=0.2,
        discount_factor=0.99,
        normalize_observations=False,
        normalize_rewards=False,
        use_gae=True,
        num_epochs=params["num_epochs"],
        debug_summaries=False,
        summarize_grads_and_vars=False,
        train_step_counter=global_step
    )

    agent.initialize()

# Get agent policies
eval_policy = agent.policy
collect_policy = agent.collect_policy

# Instantiate Replay Buffer, which holds the sampled trajectories.
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=collect_env.batch_size,
    max_length=params["replay_buffer_capacity"]
)

# Instantiate driver for data collection
environment_steps_metric = tf_metrics.EnvironmentSteps()
environment_episodes_metric = tf_metrics.NumberOfEpisodes()
train_avg_return = tf_metrics.AverageReturnMetric(
    buffer_size=collect_env.n_envs, batch_size=collect_env.n_envs
)
step_metrics = [
    environment_episodes_metric,
    environment_steps_metric,
]
train_metrics = step_metrics + [
    train_avg_return,
    tf_metrics.MinReturnMetric(buffer_size=collect_env.n_envs, batch_size=collect_env.n_envs),
    tf_metrics.MaxReturnMetric(buffer_size=collect_env.n_envs, batch_size=collect_env.n_envs),
    tf_metrics.AverageEpisodeLengthMetric(buffer_size=collect_env.n_envs, batch_size=collect_env.n_envs),
]
collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    collect_env,
    collect_policy,
    observers=[replay_buffer.add_batch] + train_metrics,
    num_episodes=collect_env.n_envs
)

# Define checkpointer to save policy
ckpt_dir = os.path.join(train_dir, "ckpt")
saved_model_dir = os.path.join(train_dir, "policy_saved_model")

train_checkpointer = common.Checkpointer(
    ckpt_dir=ckpt_dir,
    max_to_keep=params["ckpt_num"],
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'),
    global_step=global_step
)

policy_checkpointer = common.Checkpointer(
    ckpt_dir=os.path.join(train_dir, 'policy'),
    policy=eval_policy,
    global_step=global_step,
)

saved_model = policy_saver.PolicySaver(eval_policy, train_step=global_step)
train_checkpointer.initialize_or_restore()

# Create directory to save all the DRL signals generated
if params["save_DRLtxt"]:
    if not os.path.exists("./DRLsignals"):
        os.mkdir("DRLsignals")

# --------------------------- Training / Evaluation ---------------------------
with tf.compat.v2.summary.record_if(
    lambda: tf.math.equal(global_step % params["summary_interval"], 0)
):
    def train_step():
        trajectories = replay_buffer.gather_all()
        return agent.train(experience=trajectories)

    if params["use_tf_functions"]:
        collect_driver.run = common.function(collect_driver.run, autograph=False)
        agent.train = common.function(agent.train, autograph=False)
        train_step = common.function(train_step)

    if params["mode"] == "train":
        collect_time = 0
        train_time = 0
        timed_at_step = agent.train_step_counter.numpy()

        # Write parameter files to Tensorboard and plots in train directory
        tf.summary.text("params", params_str(params), step=global_step.numpy())
        history = History(train_dir)

        # Train loop
        logger.info(f"{bcolors.BOLD}Starting training loop!{bcolors.ENDC}")
        logger.info(f"Current training global step: {timed_at_step}\n")
        while environment_episodes_metric.result() < params["num_episodes"]:
            logger.info(f"{bcolors.OKBLUE}Collect environment running{bcolors.ENDC}")
            global_step_val = global_step.numpy()
            start_time = time.time()
            collect_env.start(
                new_ensamble=(global_step_val == 0),
                restart_file=(params["restart_file"] if global_step_val > 0 else 1),
                global_step=global_step_val
            )
            collect_driver.run()
            collect_env.stop()
            collect_time += time.time() - start_time

            start_time = time.time()
            total_loss, _ = train_step()
            replay_buffer.clear()
            train_time += time.time() - start_time

            for train_metric in train_metrics:
                train_metric.tf_summaries(train_step=global_step, step_metrics=step_metrics)

            if global_step_val % params["ckpt_interval"] == 0:
                logger.info(f"Saving checkpoint to: {ckpt_dir}")
                train_checkpointer.save(global_step_val)
                policy_checkpointer.save(global_step_val)
                saved_model_path = os.path.join(
                    saved_model_dir, 'policy_' + f'{global_step_val:09d}'
                )
                saved_model.save(saved_model_path)

            if global_step_val % params["log_interval"] == 0:
                logger.info(f"{bcolors.OKCYAN}Training stats:{bcolors.ENDC}")
                logger.info('step = %d, loss = %f', global_step_val, total_loss)
                steps_per_hour = (
                    (agent.train_step_counter.numpy() - timed_at_step)
                    / (collect_time + train_time) * 3600
                )
                logger.info('%.4f steps/hour', steps_per_hour)
                logger.info('collect_time = %.4f, train_time = %.4f', collect_time, train_time)
                with tf.compat.v2.summary.record_if(True):
                    tf.compat.v2.summary.scalar(
                        name='global_steps_per_hour',
                        data=steps_per_hour,
                        step=global_step
                    )

                logger.info(f"Episodes: {environment_episodes_metric.result().numpy()}")
                logger.info(f"Global training steps: {agent.train_step_counter.numpy()}")
                logger.info(f"Environment steps: {environment_steps_metric.result().numpy()}")
                logger.info(f"Average reward: {numpy_str(train_avg_return.result().numpy())}")

                history.plot()
                logger.info("Plotting training metrics done!\n")

                timed_at_step = agent.train_step_counter.numpy()
                collect_time = 0
                train_time = 0

            # Store the DRL signals generated during the episode
            if params["save_DRLtxt"]:
                for env_idx in range(params["cfd_n_envs"]):
                    out_dir = f"./DRLsignals/output_{env_idx}_{int(global_step_val)}"
                    if not os.path.exists(out_dir):
                        os.mkdir(out_dir)

                    os.replace(
                        f"output_{env_idx}/ClCd.txt",
                        f"{out_dir}/ClCd.txt"
                    )
                    os.replace(
                        f"output_{env_idx}/ClCd_avg.txt",
                        f"{out_dir}/ClCd_avg.txt"
                    )
                    os.replace(
                        f"output_{env_idx}/control_reward.txt",
                        f"{out_dir}/control_reward.txt"
                    )
                    os.replace(
                        f"output_{env_idx}/control_action.txt",
                        f"{out_dir}/control_action.txt"
                    )
                    os.replace(
                        f"output_{env_idx}/smooth_control_action.txt",
                        f"{out_dir}/smooth_control_action.txt"
                    )

        logger.info(f"{bcolors.BOLD}Ended training loop!{bcolors.ENDC}\n")

    elif params['mode'] == "eval":
        # Init environment
        eval_py_env = SodEnvCyl(
            exp,
            db,
            hosts,
            "sod2d",
            cwd,
            cfd_n_envs=1,
            mode="eval",
            **env_params,
        )
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

        eval_dir = os.path.join(cwd, "eval")
        eval_summary_writer = tf.summary.create_file_writer(eval_dir, flush_millis=1000)
        eval_avg_return = tf_metrics.AverageReturnMetric(
            buffer_size=eval_env.n_envs, batch_size=eval_env.n_envs
        )
        eval_metrics = [
            eval_avg_return,
            tf_metrics.MinReturnMetric(buffer_size=eval_env.n_envs, batch_size=eval_env.n_envs),
            tf_metrics.MaxReturnMetric(buffer_size=eval_env.n_envs, batch_size=eval_env.n_envs),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=eval_env.n_envs, batch_size=eval_env.n_envs),
        ]
        history = History(eval_dir)

        logger.info(f"{bcolors.OKBLUE}Evaluation environment running{bcolors.ENDC}")
        logger.info(f"{bcolors.OKCYAN}  - Agent trained with {global_step.numpy()} MARL episodes.{bcolors.ENDC}")

        eval_env.start(
            new_ensamble=True,
            restart_file=1,
            global_step=global_step.numpy()
        )
        metric_utils.eager_compute(
            eval_metrics,
            eval_env,
            eval_policy,
            num_episodes=1,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix="Metrics",
        )
        eval_env.stop()

        logger.info(f"{bcolors.OKCYAN}Evaluation stats:{bcolors.ENDC}")
        logger.info(f"Average reward (eval): {numpy_str(eval_avg_return.result().numpy())}")
        history.plot()
        logger.info("Plotting evaluation metrics done!")

    else:
        logger.info(f"Mode = {params['mode']} not recognised. Aborting simulation.")

# Kill database
exp.stop(db)
time.sleep(2.0)
