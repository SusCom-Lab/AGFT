import numpy as np
import pickle
import json
from pathlib import Path
from typing import Optional
from datetime import datetime
from .logger import setup_logger

logger = setup_logger(__name__)

class LinUCB:
    """
    Hybrid/Global LinUCB - 所有动作共享一套参数
    使用one-hot编码拼接动作信息
    """
    def __init__(self, n_features: int, n_actions: int, alpha: float = 3.0,
                 lambda_reg: float = 0.1,
                 model_dir: str = "data/models", auto_load: bool = True):
        self.n_features = n_features   # 环境特征维度
        self.n_actions = n_actions     # 动作数量
        self.d = n_features + n_actions  # 总维度 = 环境特征 + one-hot动作
        self.alpha = alpha
        self.initial_alpha = alpha
        self.lambda_reg = lambda_reg
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 全局共享的A和b矩阵
        self.A = self.lambda_reg * np.eye(self.d, dtype=np.float32)
        self.b = np.zeros(self.d, dtype=np.float32)

        # 统计信息
        self.action_counts = [0] * n_actions
        self.total_rounds = 0
        self.total_reward = 0.0
        self.reward_history = []
        self.last_action = None  # 记录上一次的动作（用于计算切换成本）

        # 用于冷启动的随机排列
        self._cold_start_permutation = None

        # 模型元数据
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'version': '3.0-hybrid'
        }

        logger.info(f"🤖 初始化Hybrid LinUCB:")
        logger.info(f"   环境特征: {n_features}维")
        logger.info(f"   动作数量: {n_actions}个")
        logger.info(f"   总维度: {self.d}维")
        logger.info(f"   初始α: {alpha}")
        logger.info(f"   正则化λ: {lambda_reg}")

        if auto_load:
            self.load_model()
        else:
            logger.info("🆕 跳过模型加载，从全新状态开始")

    def _create_context_with_action(self, base_features: np.ndarray, action: int) -> np.ndarray:
        """将环境特征和动作one-hot编码拼接"""
        # One-hot编码动作
        action_one_hot = np.zeros(self.n_actions, dtype=np.float32)
        action_one_hot[action] = 1.0
        
        # 拼接
        context = np.concatenate([base_features, action_one_hot])
        
        assert context.shape[0] == self.d, \
            f"拼接后维度错误: {context.shape[0]} vs {self.d}"
        
        return context

    def select_action(self, base_features: np.ndarray) -> int:
        """选择动作 - 使用UCB策略"""
        # 冷启动阶段：前n_actions*3轮，随机顺序探索每个动作
        explore_rounds = self.n_actions * 3
        if self.total_rounds < explore_rounds:
            # 初始化随机排列
            if self._cold_start_permutation is None or len(self._cold_start_permutation) < explore_rounds:
                # 生成3轮完整的随机排列
                self._cold_start_permutation = []
                for _ in range(3):
                    self._cold_start_permutation.extend(np.random.permutation(self.n_actions))
            
            selected = self._cold_start_permutation[self.total_rounds]
            logger.info(f"🎲 [冷启动探索] 轮次 {self.total_rounds + 1}/{explore_rounds}, 选择动作 {selected}")
            return selected

        # 计算当前参数估计
        try:
            theta = np.linalg.solve(self.A, self.b)
        except np.linalg.LinAlgError:
            logger.warning("⚠️ A矩阵奇异，使用伪逆")
            theta = np.linalg.pinv(self.A) @ self.b

        # 计算每个动作的UCB值
        ucb_values = []
        for action in range(self.n_actions):
            # 构造完整特征
            x = self._create_context_with_action(base_features, action)
            
            # 预测值
            pred = theta.dot(x)
            
            # 置信区间（使用Cholesky分解加速）
            try:
                L = np.linalg.cholesky(self.A)
                v = np.linalg.solve(L, x)
                confidence = self.alpha * np.sqrt(np.dot(v, v))
            except np.linalg.LinAlgError:
                # 如果Cholesky分解失败，使用标准方法
                A_inv_x = np.linalg.solve(self.A, x)
                confidence = self.alpha * np.sqrt(x.dot(A_inv_x))
            
            ucb = pred + confidence
            ucb_values.append(ucb)
            
            logger.debug(f"  动作{action}: pred={pred:.3f}, conf={confidence:.3f}, UCB={ucb:.3f}")

        # 选择UCB最大的动作（有并列时随机选择）
        max_ucb = max(ucb_values)
        candidates = [i for i, v in enumerate(ucb_values) if abs(v - max_ucb) < 1e-9]
        selected = np.random.choice(candidates)
        
        logger.info(f"🎯 选择动作 {selected}, UCB值: {ucb_values[selected]:.3f}")
        
        return selected

    def update(self, action: int, base_features: np.ndarray, reward: float):
        """更新模型参数"""
        # 构造完整特征
        x = self._create_context_with_action(base_features, action)
        
        # 更新A和b（不使用衰减）
        self.A += np.outer(x, x)
        self.b += reward * x

        # 更新统计信息
        self.action_counts[action] += 1
        self.total_rounds += 1
        self.total_reward += reward
        self.reward_history.append(reward)
        if len(self.reward_history) > 1000:
            self.reward_history.pop(0)
        
        self.last_action = action

        # α衰减策略：前3*K轮保持高探索，之后缓慢衰减
        if self.total_rounds <= self.n_actions * 3:
            # 冷启动阶段保持高探索
            self.alpha = self.initial_alpha
        else:
            # 使用更缓慢的衰减
            rounds_after_explore = self.total_rounds - self.n_actions * 3
            self.alpha = max(0.5, self.initial_alpha / np.sqrt(1.0 + 0.01 * rounds_after_explore))
        
        logger.info(f"📈 更新模型: 动作={action}, 奖励={reward:.6f}, α={self.alpha:.3f}, 总轮次={self.total_rounds}")

        # 定期保存模型
        if self.total_rounds % 10 == 0:
            self.save_model()

    def save_model(self, filename: Optional[str] = None):
        """保存模型"""
        if filename is None:
            filename = f"linucb_hybrid_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        filepath = self.model_dir / filename
        
        # 更新元数据
        self.metadata['updated_at'] = datetime.now().isoformat()
        self.metadata['total_rounds'] = self.total_rounds
        self.metadata['total_reward'] = self.total_reward
        self.metadata['avg_reward'] = self.total_reward / max(self.total_rounds, 1)
        
        # 打包模型数据
        model_data = {
            'A': self.A.astype(np.float32),
            'b': self.b.astype(np.float32),
            'n_features': self.n_features,
            'n_actions': self.n_actions,
            'd': self.d,
            'alpha': self.alpha,
            'initial_alpha': self.initial_alpha,
            'lambda_reg': self.lambda_reg,
            'action_counts': self.action_counts,
            'total_rounds': self.total_rounds,
            'total_reward': self.total_reward,
            'reward_history': self.reward_history[-100:],  # 只保存最近100个
            'last_action': self.last_action,
            'metadata': self.metadata
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"💾 模型已保存到: {filepath}")
            
            # 创建软链接指向最新模型
            latest_path = self.model_dir / "latest_model.pkl"
            if latest_path.exists():
                latest_path.unlink()
            latest_path.symlink_to(filepath.name)
            
            # 保存元数据JSON
            meta_path = self.model_dir / "model_metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"保存模型失败: {e}")

    def load_model(self, filename: Optional[str] = None):
        """加载模型"""
        if filename is None:
            # 尝试加载最新模型
            latest_path = self.model_dir / "latest_model.pkl"
            if latest_path.exists():
                filepath = latest_path
            else:
                # 查找最新的模型文件
                model_files = list(self.model_dir.glob("linucb_hybrid_model_*.pkl"))
                if not model_files:
                    logger.info("📂 没有找到已保存的模型，将从头开始学习")
                    return False
                filepath = max(model_files, key=lambda p: p.stat().st_mtime)
        else:
            filepath = self.model_dir / filename
            
        if not filepath.exists():
            logger.warning(f"模型文件不存在: {filepath}")
            return False
            
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            # 兼容性检查
            if model_data.get('version', '').startswith('3.0'):
                # 新版本模型
                if (model_data['n_features'] != self.n_features or 
                    model_data['n_actions'] != self.n_actions):
                    logger.warning("⚠️ 加载的模型参数与当前配置不符")
                    return False
            else:
                # 旧版本模型，无法兼容
                logger.warning("⚠️ 模型版本过旧，无法加载")
                return False
                
            # 恢复参数
            self.A = model_data['A']
            self.b = model_data['b']
            self.d = model_data['d']
            self.alpha = model_data['alpha']
            self.initial_alpha = model_data['initial_alpha']
            self.lambda_reg = model_data['lambda_reg']
            self.action_counts = model_data['action_counts']
            self.total_rounds = model_data['total_rounds']
            self.total_reward = model_data['total_reward']
            self.reward_history = model_data.get('reward_history', [])
            self.last_action = model_data.get('last_action', None)
            self.metadata = model_data['metadata']
            
            logger.info(f"✅ 模型已加载: {filepath}")
            logger.info(f"   总轮次: {self.total_rounds}")
            logger.info(f"   平均奖励: {self.total_reward / max(self.total_rounds, 1):.3f}")
            logger.info(f"   当前α: {self.alpha:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False

    def get_model_stats(self) -> dict:
        """获取模型统计信息"""
        stats = {
            'total_rounds': self.total_rounds,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(self.total_rounds, 1),
            'action_counts': self.action_counts,
            'current_alpha': self.alpha,
            'recent_rewards': self.reward_history[-20:] if self.reward_history else [],
            'last_action': self.last_action,
            'metadata': self.metadata
        }
        
        # 计算动作分布熵
        if sum(self.action_counts) > 0:
            probs = np.array(self.action_counts) / sum(self.action_counts)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            stats['action_entropy'] = entropy
            stats['effective_actions'] = np.exp(entropy)
        
        return stats