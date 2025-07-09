#!/bin/bash

# vLLM MAB Data and Logs Cleanup Script
# Sets proper permissions and cleans up data/logs directories for user colin

echo "🧹 vLLM MAB Cleanup Script"
echo "=========================="

# Function to safely remove files and directories
cleanup_directory() {
    local dir=$1
    local desc=$2
    
    if [ -d "$dir" ]; then
        echo "📁 Cleaning $desc ($dir)..."
        
        # Change ownership to colin recursively
        sudo chown -R colin:colin "$dir" 2>/dev/null || {
            echo "⚠️  需要sudo权限来更改 $dir 的所有权"
            echo "   请运行: sudo chown -R colin:colin $dir"
            return 1
        }
        
        # Set proper permissions
        chmod -R 755 "$dir"
        
        # Remove all contents
        find "$dir" -mindepth 1 -delete 2>/dev/null || {
            echo "❌ 无法删除 $dir 中的某些文件"
            return 1
        }
        
        echo "✅ $desc 清理完成"
    else
        echo "⚠️  目录 $dir 不存在"
    fi
}

# Main cleanup function
main() {
    echo "当前用户: $(whoami)"
    echo "工作目录: $(pwd)"
    echo ""
    
    # Cleanup logs directory (now with GPU classification)
    if [ -d "logs" ]; then
        echo "📁 清理GPU分类日志目录 (logs)..."
        
        # Change ownership recursively
        sudo chown -R colin:colin logs 2>/dev/null || {
            echo "⚠️  需要sudo权限来更改 logs 的所有权"
            echo "   请运行: sudo chown -R colin:colin logs"
        }
        
        # Set proper permissions
        chmod -R 755 logs
        
        # List GPU model directories and their log files before cleanup
        echo "📊 发现的GPU型号目录:"
        find logs -mindepth 1 -maxdepth 1 -type d -exec basename {} \; 2>/dev/null | sort | while read gpu_dir; do
            log_count=$(find "logs/$gpu_dir" -name "*.log" 2>/dev/null | wc -l)
            echo "   - $gpu_dir ($log_count 个日志文件)"
        done
        
        # Remove all contents
        find logs -mindepth 1 -delete 2>/dev/null || {
            echo "❌ 无法删除 logs 中的某些文件"
            return 1
        }
        
        echo "✅ GPU分类日志目录清理完成"
    else
        echo "⚠️  日志目录不存在"
    fi
    
    # Cleanup data directory (keeping structure but removing files)
    if [ -d "data" ]; then
        echo "📁 清理数据目录 (data)..."
        
        # Change ownership of data directory
        sudo chown -R colin:colin data 2>/dev/null || {
            echo "⚠️  需要sudo权限来更改 data 的所有权"
            echo "   请运行: sudo chown -R colin:colin data"
        }
        
        # Clean models directory but keep the directory structure
        if [ -d "data/models" ]; then
            find data/models -name "*.pkl" -delete 2>/dev/null
            find data/models -name "*.json" -delete 2>/dev/null
            echo "✅ 模型文件已清理"
        fi
        
        # Clean analysis directory
        if [ -d "data/analysis" ]; then
            find data/analysis -name "*.png" -delete 2>/dev/null
            find data/analysis -name "*.jpg" -delete 2>/dev/null
            echo "✅ 分析图片已清理"
        fi
        
        echo "✅ 数据目录清理完成"
    else
        echo "⚠️  数据目录不存在"
    fi
    
    echo ""
    echo "🎉 清理完成！"
    echo ""
    echo "📊 清理后的目录状态:"
    echo "==================="
    ls -la logs data 2>/dev/null || echo "目录已被清空或不存在"
}

# Show help
show_help() {
    echo "用法: ./cleanup.sh [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  -f, --force    强制清理（不提示确认）"
    echo "  --logs-only    仅清理日志目录"
    echo "  --data-only    仅清理数据目录"
    echo ""
    echo "示例:"
    echo "  ./cleanup.sh              # 交互式清理所有"
    echo "  ./cleanup.sh --force      # 强制清理所有"
    echo "  ./cleanup.sh --logs-only  # 仅清理日志"
}

# Parse command line arguments
FORCE=false
LOGS_ONLY=false
DATA_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        --logs-only)
            LOGS_ONLY=true
            shift
            ;;
        --data-only)
            DATA_ONLY=true
            shift
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# Confirmation prompt
if [ "$FORCE" = false ]; then
    echo "⚠️  这将删除以下目录中的所有文件:"
    if [ "$DATA_ONLY" = false ]; then
        echo "   - logs/ (所有日志文件)"
    fi
    if [ "$LOGS_ONLY" = false ]; then
        echo "   - data/models/ (所有模型文件)"
        echo "   - data/analysis/ (所有分析图片)"
    fi
    echo ""
    read -p "确认继续？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ 操作已取消"
        exit 1
    fi
fi

# Execute cleanup based on options
if [ "$LOGS_ONLY" = true ]; then
    cleanup_directory "logs" "日志目录"
elif [ "$DATA_ONLY" = true ]; then
    cleanup_directory "data" "数据目录"
else
    main
fi