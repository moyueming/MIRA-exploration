import argparse
import os
import json
import numpy as np

from rl.A3C_2_actors.operation_actor import OperationActor
from rl.A3C_2_actors.pipeline_environment import PipelineEnvironment
from rl.A3C_2_actors.set_actor import SetActor
from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets

parser = argparse.ArgumentParser()
# 动态传入我们训练出的模型文件夹名称
parser.add_argument('--name', type=str, required=True, help="Name of the trained model folder in saved_models/")
args = parser.parse_args()

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

class TripartiteAgentRunner:
    def __init__(self, model_name):
        self.model_name = model_name
        self.base_path = f"./saved_models/{model_name}"
        
        # 严谨的学术做法：从训练时的 info.json 读取当时的环境参数
        with open(f"{self.base_path}/info.json") as f:
            self.train_args = json.load(f)
            
        self.episode_steps = 250 if self.train_args.get("mode", "scattered") == "scattered" else 25
        self.steps = self.train_args.get("lstm_steps", 5)
        
        data_folder = "./app/data/"
        self.pipeline = PipelineWithPrecalculatedSets(
            "sdss", ["galaxies"], data_folder=data_folder, discrete_categories_count=10, min_set_size=10, 
            exploration_columns=["galaxies.u", "galaxies.g", "galaxies.r", "galaxies.i", "galaxies.z", "galaxies.petroRad_r", "galaxies.redshift"])
            
        # 初始化验证环境
        self.env = PipelineEnvironment(
            self.pipeline, mode=self.train_args.get("mode", "scattered"), episode_steps=self.episode_steps, 
            operators=self.train_args.get("operators", ["by_facet", "by_superset", "by_neighbors", "by_distribution"]))

        self.set_state_dim = self.env.set_state_dim
        self.operation_state_dim = self.env.operation_state_dim
        self.set_action_dim = self.env.set_action_space.n
        self.operation_action_dim = self.env.operation_action_space.n
        
        # 加载训练好的 Actor 网络权重 (注意路径指向 current)
        self.set_actor = SetActor(
            self.set_state_dim, self.set_action_dim, self.steps, 0, self.model_name, 
            model_path=f"{self.base_path}/current/set_actor")
        self.operation_actor = OperationActor(
            self.operation_state_dim, self.operation_action_dim, self.steps, 0, self.model_name, 
            model_path=f"{self.base_path}/current/operation_actor")

    # 复用 Coherency 的数学工具
    def compute_cosine_similarity(self, vec_a, vec_b):
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(vec_a, vec_b) / (norm_a * norm_b)

    def run(self, times=10):
        results = []
        for i in range(times):
            print(f"--- Running Evaluation Episode: {i+1}/{times} ---")
            done = False
            set_action_steps = [[0.0] * self.set_state_dim] * self.steps
            operation_action_steps = [[0.0] * self.operation_state_dim] * self.steps
            
            set_state = self.env.reset()
            set_action_steps.pop(0)
            set_action_steps.append(set_state)
            
            episode_history_phi = [np.array(set_state).flatten()]
            step_rewards_log = []
            
            while not done:
                # 策略推理：Actor 给出动作概率
                probs = self.set_actor.model.predict(np.array(set_action_steps).reshape((1, self.steps, self.set_state_dim)))
                probs = self.env.fix_possible_set_action_probs(probs[0])
                set_action = 0 if all(np.isnan(x) for x in probs) else np.random.choice(self.set_action_dim, p=probs)

                operation_state = self.env.get_operation_state(set_action)
                operation_action_steps.pop(0)
                operation_action_steps.append(operation_state)
                
                probs = self.operation_actor.model.predict(np.array(operation_action_steps).reshape((1, self.steps, self.operation_state_dim)))
                probs = self.env.fix_possible_operation_action_probs(set_action, probs[0])
                operation_action = self.env.get_random_operation(set_action) if np.isnan(probs[0]) else np.random.choice(self.operation_action_dim, p=probs)

                # 环境步进
                next_set_state, env_reward, done, _ = self.env.step(set_action, operation_action)
                
                # ==========================================
                # 在线评估：重新解算三元物理量用于日志记录
                # ==========================================
                r_int = float(np.squeeze(env_reward)) # Interestingness
                
                phi_s_t = np.array(set_state).flatten()
                phi_s_next = np.array(next_set_state).flatten()

                r_coh = float(self.compute_cosine_similarity(phi_s_t, phi_s_next)) # Coherency
                
                # Diversity
                distances = [np.linalg.norm(phi_s_next - h) for h in episode_history_phi]
                min_dist = np.min(distances)
                r_div = float(1.0 - np.exp(-0.1 * min_dist))
                    
                episode_history_phi.append(phi_s_next)
                
                # 封存该步的三元奖励信息
                step_rewards_log.append({
                    "r_int": r_int,
                    "r_coh": r_coh,
                    "r_div": r_div
                })

                set_state = next_set_state
                next_set_action_steps = set_action_steps.copy()
                next_set_action_steps.pop(0)
                next_set_action_steps.append(next_set_state)
                set_action_steps = next_set_action_steps

            # 将三元奖励追加到环境原本的 episode_info 中
            for idx, info in enumerate(self.env.episode_info):
                info.update(step_rewards_log[idx])
            results.append(self.env.episode_info)
            
        return results

if __name__ == "__main__":
    runner = TripartiteAgentRunner(args.name)
    eval_results = runner.run(times=10) # 进行 10 次完整 Episode 评估
    
    # 结果持久化，方便你后续导入 Jupyter Notebook 画图
    save_path = f"./saved_models/{args.name}/evaluation_results.json"
    with open(save_path, 'w') as f:
        json.dump(eval_results, f, indent=1, default=np_encoder)
    print(f"Evaluation finished. Results saved to {save_path}")