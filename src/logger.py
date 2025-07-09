import logging
import sys
from datetime import datetime
from pathlib import Path

class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
    }
    RESET = '\033[0m'
    
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        return super().format(record)

def setup_logger(name, log_file=None, console_level=logging.INFO, file_level=logging.DEBUG):
    """设置日志器 - 支持控制台和文件不同级别"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # 防止重复添加处理器
    if logger.handlers:
        return logger
    
    # 防止向根logger传播（避免双重输出）
    logger.propagate = False
    
    # 控制台处理器（带颜色，简化输出）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = ColoredFormatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # 可选添加过滤器，排除详细记录日志（默认启用简化模式）
    class ConsoleFilter(logging.Filter):
        def filter(self, record):
            # 过滤掉详细记录的日志和频繁的添加频率日志（不在控制台显示）
            msg = record.getMessage()
            # 过滤条件：详细轮次记录、JSON数据、频繁的频率添加日志
            return not (msg.startswith('🔍 轮次') or 
                       msg.startswith('📋 JSON数据:') or
                       msg.startswith('➕ 添加新频率:') or
                       msg.startswith('🆕 初始化频率'))
    
    # 默认启用简化模式，可通过参数控制
    console_handler.addFilter(ConsoleFilter())
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定）- 记录所有详细信息
    if log_file:
        # 支持传入完整路径或仅文件名
        if isinstance(log_file, Path) or '/' in str(log_file):
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            Path("logs").mkdir(exist_ok=True)
            log_path = Path("logs") / log_file
            
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger