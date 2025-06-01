#!/usr/bin/env bash
# run_all.sh
set -e                       # 任何命令非零退出时立即终止脚本

profiles=(p1 p2 p3 p4 p5)    # 五个 profile
ws=(1 1.5 2)                 # w 取值
ks=(1 0 -1)                  # k 取值，按示例顺序

for profile in "${profiles[@]}"; do
  echo "=== Running profile ${profile} ==="

  # -------- cfl task: 3 w × 3 k = 9 次，共 45 条（5 profiles）
  for w in "${ws[@]}"; do
    for k in "${ks[@]}"; do
      python dummy_sols/burgers_1d.py --profile "${profile}" --task cfl --k "${k}" --w "${w}"
    done
  done

  # -------- k task: 3 次（w 变化）
  for w in "${ws[@]}"; do
    python dummy_sols/burgers_1d.py --profile "${profile}" --task k --w "${w}"
  done

  # -------- w task: 3 次（k 变化）
  for k in "${ks[@]}"; do
    python dummy_sols/burgers_1d.py --profile "${profile}" --task w --k "${k}"
  done
done

echo "✅ 全部任务完成！"
