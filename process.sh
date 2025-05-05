#!/bin/bash

# 检查是否传入正确的参数
#if [ $# -lt 2 ]; then
##  echo "Usage: \$0 <start_date> <end_date> <stock_start> <stock_end>"
#  exit 1
#fi

# 参数传递
start_date=\$1
end_date=\$2

stock_start=\$3
stock_end=\$4

# 设置文件夹路径
base_dir="quote"

# 设置数据类型
types=("trade_in_hour" "snap_in_hour" "order_in_hour")

# 生成股票代码列表
stock_codes=()
for i in $(seq -f "%06g" $stock_start $stock_end); do
  stock_codes+=("sh$i")
done

# 迭代日期范围
current_date=$start_date
while [[ "$current_date" <= "$end_date" ]]; do
  # 遍历每个股票代码和类型
  for stock_code in "${stock_codes[@]}"; do
    file_paths=()  # 用来存储当前三个文件路径
    # 为每种类型构建文件路径
    for type in "${types[@]}"; do
      file_path="$base_dir/$type/$current_date/$stock_code.json"

      if [ -f "$file_path" ]; then
        file_paths+=("$file_path")
      else
        echo "File not found: $file_path"
      fi

      # 如果已经收集到三个文件路径，调用 Python 脚本
      if [ ${#file_paths[@]} -eq 3 ]; then
        echo "Processing files: ${file_paths[@]}"
        python process_file.py "${file_paths[@]}"
        file_paths=()  # 清空文件路径列表，准备下一个批次
      fi

    done
  done
  # 获取下一个日期（可以使用 date 命令）
  current_date=$(date -I -d "$current_date + 1 day")
done

echo "Processing completed."
