import argparse
import os
import json
import random
import traceback
import csv
from datetime import datetime

import numpy as np
import tensorflow as tf
import ray
os.environ.setdefault("WANDB_MODE", "offline")
import wandb

from .critic import Critic
from .operation_actor import OperationActor
from .pipeline_environment import PipelineEnvironment
from .set_actor import SetActor

tf.keras.backend.set_floatx('float64')

now = datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=20)
parser.add_argument('--actor_lr', type=float, default=0.00003)
parser.add_argument('--critic_lr', type=float, default=0.00003)
parser.add_argument('--workers', type=int, default=12)
parser.add_argument('--lstm_steps', type=int, default=5)

# ==========================================
# 融合架构核心参数
# ==========================================
# 挂载原始论文 Target Set 以激活外部主线奖励
parser.add_argument('--target_set', type=str, default=None)
parser.add_argument('--target_seed', type=int, default=None)
parser.add_argument('--target_samples_per_file', type=int, default=100)

# Original paper familiarity-curiosity reward.
# Default 0.25 corresponds to 75FAM-25CUR.
parser.add_argument('--counter_curiosity_ratio', type=float, default=0.25)

parser.add_argument('--notes', type=str, default="")
parser.add_argument('--mode', type=str, default="scattered")
parser.add_argument('--name', type=str, default="")
parser.add_argument('--resume', action='store_true')
parser.add_argument('--resume_step', type=int, default=None)
parser.add_argument('--operators', nargs='+', type=str,
                    default=["by_facet", "by_superset", "by_neighbors", "by_distribution"])
args = parser.parse_args()

if args.resume_step != None:
    args.resume = True

if args.name == "":
    fam_ratio = int(round((1.0 - args.counter_curiosity_ratio) * 100))
    cur_ratio = int(round(args.counter_curiosity_ratio * 100))
    args.name = f"{args.mode}-paper-a3c-{fam_ratio}fam-{cur_ratio}cur-lstm-{args.lstm_steps}-alr-{args.actor_lr}-clr-{args.critic_lr}-{now.strftime('%m%d%Y_%H%M%S')}"

if not args.resume:
    args.id = wandb.util.generate_id()
    if not os.path.exists("saved_models/" + args.name):
        os.makedirs("saved_models/" + args.name)
    with open("saved_models/" + args.name + "/info.json", 'w') as f:
        json.dump(vars(args), f, indent=1)
else:
    with open("./saved_models/"+args.name+"/info.json") as f:
        items = json.load(f)
        for key in items.keys():
            if key != "resume" and key != "resume_step":
                setattr(args, key, items[key])

wandb.init(name=args.name, project="deep-rl-tf2", id=args.id, resume=args.resume, config=vars(args))


@ray.remote
class ParameterServer:
    def __init__(self, set_state_dim, operation_state_dim, set_action_dim, operation_action_dim, args):
        import tensorflow as tf
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)

        self.args = args
        self.episodes_done = args.resume_step if args.resume_step is not None else 0

        model_path = None
        if args.resume:
            model_path = f"./saved_models/{args.name}/{args.resume_step}/" if args.resume_step else f"./saved_models/{args.name}/current/"

        self.global_set_actor = SetActor(set_state_dim, set_action_dim, args.lstm_steps, args.actor_lr, args.name, model_path=model_path+"set_actor" if model_path else None)
        self.global_operation_actor = OperationActor(operation_state_dim, operation_action_dim, args.lstm_steps, args.actor_lr, args.name, model_path=model_path+"operation_actor" if model_path else None)
        self.global_critic = Critic(set_state_dim, args.lstm_steps, args.critic_lr, args.name, model_path=model_path+"critic" if model_path else None)
        self.global_sets_viewed = set()
        self.cumulative_extrinsic_reward = 0.0
        self.logged_exploration_state_ids = set()

    def record_episode_metrics(self, ep_ext_score, episode_set_ids):
        episode_sets = set()
        for set_id in episode_set_ids:
            try:
                parsed_set_id = int(set_id)
            except (TypeError, ValueError):
                continue
            if parsed_set_id >= 0:
                episode_sets.add(parsed_set_id)

        self.global_sets_viewed.update(episode_sets)
        self.cumulative_extrinsic_reward += float(ep_ext_score)

        episode_sets_viewed = len(episode_sets)
        cumulative_unique_sets_viewed = len(self.global_sets_viewed)
        stats = {
            "sets_viewed": episode_sets_viewed,
            "cumulative_unique_sets_viewed": cumulative_unique_sets_viewed,
            "target_efficiency": float(ep_ext_score) / max(episode_sets_viewed, 1),
            "cumulative_extrinsic_reward": self.cumulative_extrinsic_reward,
            "cumulative_target_efficiency": self.cumulative_extrinsic_reward / max(cumulative_unique_sets_viewed, 1),
        }
        stats["episode"] = self.increment_and_check_save()
        return stats

    def record_exploration_logs(self, episode, trace_rows, state_rows, context=None):
        context = context or {}

        if trace_rows:
            trace_file = f"{self.args.name}_exploration_trace.csv"
            file_exists = os.path.isfile(trace_file)
            with open(trace_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        'episode',
                        'agent_id',
                        'step',
                        'set_id',
                        'step_extrinsic_reward',
                        'step_interestingness',
                        'operator',
                        'parameter',
                        'input_set_id',
                        'operation_action',
                        'bootstrap_active',
                        'escape_active',
                        'z_source',
                    ])
                for row in trace_rows:
                    writer.writerow([
                        episode,
                        row.get("agent_id", -1),
                        row.get("step", -1),
                        row.get("set_id", -1),
                        row.get("step_extrinsic_reward", 0.0),
                        row.get("step_interestingness", 0.0),
                        row.get("operator", ""),
                        row.get("parameter", ""),
                        row.get("input_set_id", -1),
                        row.get("operation_action", -1),
                        int(context.get("bootstrap_active", 0)),
                        int(context.get("escape_active", 0)),
                        context.get("z_source", ""),
                    ])

        new_state_rows = []
        for row in state_rows:
            try:
                set_id = int(row.get("set_id", -1))
            except (TypeError, ValueError):
                continue
            if set_id < 0 or set_id in self.logged_exploration_state_ids:
                continue
            state = row.get("state", [])
            if not state:
                continue
            self.logged_exploration_state_ids.add(set_id)
            new_state_rows.append((set_id, state))

        if new_state_rows:
            state_file = f"{self.args.name}_visited_set_states.csv"
            file_exists = os.path.isfile(state_file)
            with open(state_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    state_dim = len(new_state_rows[0][1])
                    writer.writerow(['set_id'] + [f'state_{i}' for i in range(state_dim)])
                for set_id, state in new_state_rows:
                    writer.writerow([set_id] + state)

        return {
            "trace_rows": len(trace_rows),
            "new_state_rows": len(new_state_rows),
        }

    def apply_gradients_and_get_weights(self, set_grads, op_grads, critic_grads):
        if set_grads: self.global_set_actor.apply_gradients(set_grads)
        if op_grads: self.global_operation_actor.apply_gradients(op_grads)
        if critic_grads: self.global_critic.apply_gradients(critic_grads)
        return self.get_weights()

    def get_weights(self):
        weights = {
            'set': self.global_set_actor.model.get_weights(),
            'op': self.global_operation_actor.model.get_weights(),
            'critic': self.global_critic.model.get_weights(),
        }
        return weights

    def increment_and_check_save(self):
        self.episodes_done += 1
        ep = self.episodes_done
        if ep != 0 and ep % 250 == 0:
            self.global_operation_actor.save_model(step=ep)
            self.global_set_actor.save_model(step=ep)
            self.global_critic.save_model(step=ep)
        return ep

    def save_final_models(self):
        self.global_operation_actor.save_model()
        self.global_set_actor.save_model()
        self.global_critic.save_model()


@ray.remote
class WorkerAgent:
    def __init__(self, pipeline, ps_handle, agentId, args, episode_steps):
        import tensorflow as tf
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)

        self.agentId = agentId
        self.ps = ps_handle
        self.args = args
        self.episode_steps = episode_steps

        self.env = PipelineEnvironment(
            pipeline,
            mode=args.mode,
            agentId=agentId,
            episode_steps=episode_steps,
            operators=args.operators,
            target_set_name=args.target_set,
            target_seed=args.target_seed,
            target_samples_per_file=args.target_samples_per_file,
        )
        self.set_state_dim = self.env.set_state_dim
        self.operation_state_dim = self.env.operation_state_dim
        self.set_action_dim = self.env.set_action_space.n
        self.operation_action_dim = self.env.operation_action_space.n
        self.steps = args.lstm_steps

        self.set_actor = SetActor(self.set_state_dim, self.set_action_dim, self.steps, args.actor_lr, args.name)
        self.operation_actor = OperationActor(self.operation_state_dim, self.operation_action_dim, self.steps, args.actor_lr, args.name)
        self.critic = Critic(self.set_state_dim, self.steps, args.critic_lr, args.name)

        # ==============================================================
        # 🔬 终身记忆库：记录 Agent 在整个训练生命周期内去过哪些节点
        # ==============================================================
        self.set_op_counters = {}

        self.sync_with_ps()

    def sync_with_ps(self):
        weights = ray.get(self.ps.get_weights.remote())
        self.set_actor.model.set_weights(weights['set'])
        self.operation_actor.model.set_weights(weights['op'])
        self.critic.model.set_weights(weights['critic'])

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done: cumulative = next_v_value
        for k in reversed(range(0, len(rewards))):
            cumulative = self.args.gamma * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets

    def list_to_batch(self, list_data):
        batch = list_data[0]
        for elem in list_data[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def compute_cosine_similarity(self, vec_a, vec_b):
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(vec_a, vec_b) / (norm_a * norm_b)

    def train_loop(self, max_episodes):
        curr_episode = 0

        while max_episodes >= curr_episode:
            set_state_batch, operation_state_batch, set_action_batch, operation_action_batch, reward_batch = [], [], [], [], []

            # 原文 baseline 评估指标容器
            ep_ext_score, ep_total_reward = 0, 0
            ep_counter_curiosity_score = 0
            episode_set_op_counters = {}

            done = False

            set_action_steps = [[0.0] * self.set_state_dim] * self.steps
            operation_action_steps = [[0.0] * self.operation_state_dim] * self.steps

            set_state = self.env.reset()
            set_action_steps.pop(0)
            set_action_steps.append(set_state)
            failed = False

            try:
                while not done:
                    probs = self.set_actor.model.predict(np.array(set_action_steps).reshape((1, self.steps, self.set_state_dim)),verbose=0)
                    probs = self.env.fix_possible_set_action_probs(probs[0])
                    set_action = 0 if all(np.isnan(x) for x in probs) else np.random.choice(self.set_action_dim, p=probs)

                    operation_state = self.env.get_operation_state(set_action)
                    operation_action_steps.pop(0)
                    operation_action_steps.append(operation_state)

                    probs = self.operation_actor.model.predict(np.array(operation_action_steps).reshape((1, self.steps, self.operation_state_dim)),verbose=0)
                    probs = self.env.fix_possible_operation_action_probs(set_action, probs[0])
                    operation_action = np.random.choice(self.operation_action_dim) if all(np.isnan(x) for x in probs) else np.random.choice(self.operation_action_dim, p=probs)

                    # 执行环境步进
                    next_set_state, env_r_ext, env_r_int_js, done, set_op_pair = self.env.step(set_action, operation_action)

                    r_ext = float(np.squeeze(env_r_ext))
                    if set_op_pair in episode_set_op_counters:
                        episode_set_op_counters[set_op_pair] += 1
                    else:
                        episode_set_op_counters[set_op_pair] = 1

                    if set_op_pair in self.set_op_counters:
                        op_counter = episode_set_op_counters[set_op_pair] + self.set_op_counters[set_op_pair]
                    else:
                        op_counter = episode_set_op_counters[set_op_pair]

                    counter_curiosity_reward = (100.0 / self.episode_steps) / op_counter

                    # 2. 状态变更防火墙 (防止彻底挂机)
                    # Original paper baseline has no coherence reward.

                    # 3. 局部空间距离多样性 (用于小步挖掘期)
                    set_state = next_set_state 

                    total_step_reward = (
                        (1.0 - self.args.counter_curiosity_ratio) * r_ext
                        + self.args.counter_curiosity_ratio * counter_curiosity_reward
                    )

                    ep_ext_score += r_ext
                    ep_counter_curiosity_score += counter_curiosity_reward
                    ep_total_reward += total_step_reward

                    next_set_action_steps = set_action_steps.copy()
                    next_set_action_steps.pop(0)
                    next_set_action_steps.append(next_set_state)

                    reward_batch.append(np.reshape(total_step_reward, [1, 1]))
                    set_state_batch.append(np.array(set_action_steps).reshape((1, self.steps, self.set_state_dim)))
                    set_action_batch.append(np.reshape(set_action, [1, 1]))
                    operation_state_batch.append(np.array(operation_action_steps).reshape((1, self.steps, self.operation_state_dim)))
                    operation_action_batch.append(np.reshape(operation_action, [1, 1]))

                    if len(set_state_batch) >= self.args.update_interval or done:
                        set_states = self.list_to_batch(set_state_batch)
                        set_actions = self.list_to_batch(set_action_batch)
                        operation_states = self.list_to_batch(operation_state_batch)
                        operation_actions = self.list_to_batch(operation_action_batch)
                        rewards = self.list_to_batch(reward_batch)

                        next_v_value = self.critic.model.predict(np.array(next_set_action_steps).reshape((1, self.steps, self.set_state_dim)),verbose=0)
                        td_targets = self.n_step_td_target(rewards, next_v_value, done)
                        advantages = td_targets - self.critic.model.predict(set_states,verbose=0)

                        try:
                            set_grads, _ = self.set_actor.get_gradients(set_states, set_actions, advantages)
                            op_grads, _ = self.operation_actor.get_gradients(operation_states, operation_actions, advantages)
                            critic_grads, _ = self.critic.get_gradients(set_states, td_targets)

                            set_grads_np = [g.numpy() for g in set_grads]
                            op_grads_np = [g.numpy() for g in op_grads]
                            critic_grads_np = [g.numpy() for g in critic_grads]

                            new_weights = ray.get(
                                self.ps.apply_gradients_and_get_weights.remote(
                                    set_grads_np, op_grads_np, critic_grads_np
                                )
                            )

                            self.set_actor.model.set_weights(new_weights['set'])
                            self.operation_actor.model.set_weights(new_weights['op'])
                            self.critic.model.set_weights(new_weights['critic'])

                        except Exception as error:
                            print(error)
                            traceback.print_tb(error.__traceback__)
                            print('Episode gradient push failed, retrying')
                            failed = True
                            done = True

                        set_state_batch, operation_state_batch, set_action_batch, operation_action_batch, reward_batch = [], [], [], [], []

                    set_action_steps = next_set_action_steps

                # 单次 Episode 跑完，输出最终的核心评价指标
                if not failed:
                    for pair, count in episode_set_op_counters.items():
                        self.set_op_counters[pair] = self.set_op_counters.get(pair, 0) + count

                    episode_set_ids = list(getattr(self.env, "sets_viewed", []))
                    metrics = ray.get(self.ps.record_episode_metrics.remote(ep_ext_score, episode_set_ids))
                    curr_episode = metrics["episode"]
                    trace_rows, state_rows = self.env.consume_exploration_logs()
                    ray.get(self.ps.record_exploration_logs.remote(
                        curr_episode,
                        trace_rows,
                        state_rows,
                        {
                            "bootstrap_active": 0,
                            "escape_active": 0,
                            "z_source": "paper_a3c",
                        },
                    ))

                    print(f'EP{curr_episode} Agent{self.agentId} | Ext_R: {ep_ext_score:.1f} | CounterCur: {ep_counter_curiosity_score:.1f}')

                    csv_file = f"{self.args.name}_paper_a3c_rewards.csv"
                    file_exists = os.path.isfile(csv_file)

                    with open(csv_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow([
                                'episode',
                                'extrinsic_reward',
                                'familiar_reward',
                                'counter_curiosity',
                                'total_reward',
                                'sets_viewed',
                                'cumulative_unique_sets_viewed',
                                'target_efficiency',
                                'cumulative_extrinsic_reward',
                                'cumulative_target_efficiency',
                            ]) 
                        writer.writerow([
                            curr_episode,
                            ep_ext_score,
                            ep_ext_score,
                            ep_counter_curiosity_score,
                            ep_total_reward,
                            metrics["sets_viewed"],
                            metrics["cumulative_unique_sets_viewed"],
                            metrics["target_efficiency"],
                            metrics["cumulative_extrinsic_reward"],
                            metrics["cumulative_target_efficiency"],
                        ])

            except Exception as error:
                print(error)
                traceback.print_tb(error.__traceback__)
                print('Episode failed, retrying')

class Agent:
    def __init__(self, env_name, pipeline=None):
        self.pipeline = pipeline
        self.env_name = env_name
        self.episode_steps = 250 if args.mode == "scattered" else 25
        self.num_workers = args.workers

        dummy_env = PipelineEnvironment(
            self.pipeline,
            mode=args.mode,
            episode_steps=self.episode_steps,
            operators=args.operators,
            target_set_name=args.target_set,
            target_seed=args.target_seed,
            target_samples_per_file=args.target_samples_per_file,
        )
        self.set_state_dim = dummy_env.set_state_dim
        self.operation_state_dim = dummy_env.operation_state_dim
        self.set_action_dim = dummy_env.set_action_space.n
        self.operation_action_dim = dummy_env.operation_action_space.n
        self.target_items = sorted(map(int, dummy_env.target_items))

        if not args.resume and len(self.target_items) > 0:
            target_snapshot_path = f"saved_models/{args.name}/target_items.json"
            with open(target_snapshot_path, "w") as f:
                json.dump(self.target_items, f, indent=1)
            print(f"Saved target snapshot with {len(self.target_items)} objects to {target_snapshot_path}")

    def train(self, max_episodes=1000):
        ps = ParameterServer.remote(self.set_state_dim, self.operation_state_dim, self.set_action_dim, self.operation_action_dim, args)
        workers = [WorkerAgent.remote(self.pipeline, ps, i, args, self.episode_steps) for i in range(self.num_workers)]
        ray.get([worker.train_loop.remote(max_episodes) for worker in workers])
        ray.get(ps.save_final_models.remote())
