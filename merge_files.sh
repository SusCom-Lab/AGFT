#!/bin/bash

# 合并项目文件脚本
# 将config.yaml和src目录下所有Python文件合并到一个txt文件中

OUTPUT_FILE="merged_project_files.txt"

# 清空输出文件
> "$OUTPUT_FILE"

echo "开始合并文件..."

# 合并config.yaml
if [ -f "config/config.yaml" ]; then
    echo "=== config/config.yaml ===" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    cat "config/config.yaml" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "已添加: config/config.yaml"
fi

# 合并src目录下所有Python文件
for file in src/*.py; do
    if [ -f "$file" ]; then
        echo "=== $file ===" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        cat "$file" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        echo "已添加: $file"
    fi
done

echo "合并完成! 输出文件: $OUTPUT_FILE"
echo "文件大小: $(wc -l < "$OUTPUT_FILE") 行"