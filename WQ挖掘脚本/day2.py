from machine_lib import * 
from datetime import datetime, timedelta

s = login()

# 自动读取 Day1 的运行日期
day1_state = ManageState.get_progress("day1")
if not day1_state['date']:
    raise ValueError("Day 1 has not been run or date not recorded.")

start_date_str = day1_state['date']
# 计算结束日期 (Day1 + 1天)
start_dt = datetime.strptime(str(datetime.now().year) + "-" + start_date_str, "%Y-%m-%d")
end_date_str = (start_dt + timedelta(days=1)).strftime("%m-%d")

print(f"Fetching Day 1 alphas from {start_date_str} to {end_date_str}")

fo_tracker = get_alphas(start_date_str, end_date_str, 1.0, 0.7, "USA", 100, "track")
fo_layer = prune(fo_tracker, 'anl4', 5)

so_alpha_list = []
group_ops = ["group_neutralize", "group_rank", "group_zscore"]

for expr, decay in fo_layer:
    for alpha in get_group_second_order_factory([expr], group_ops, "USA"):
        so_alpha_list.append((alpha, decay))

random.seed(42)
random.shuffle(so_alpha_list)

so_pools = load_task_pool_single(so_alpha_list, 3)

# 自动处理断点，状态键名为 "day2"
single_simulate(so_pools, 'SUBINDUSTRY', 'USA', 'TOP3000', "day2")