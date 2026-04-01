from machine_lib import * 
from datetime import datetime, timedelta

s = login()

# 自动读取 Day2 的运行日期
day2_state = ManageState.get_progress("day2")
if not day2_state['date']:
    raise ValueError("Day 2 has not been run or date not recorded.")

start_date_str = day2_state['date']
start_dt = datetime.strptime(str(datetime.now().year) + "-" + start_date_str, "%Y-%m-%d")
end_date_str = (start_dt + timedelta(days=1)).strftime("%m-%d")

print(f"Fetching Day 2 alphas from {start_date_str} to {end_date_str}")

so_tracker = get_alphas(start_date_str, end_date_str, 1.3, 1.0, "USA", 200, "track")
so_layer = prune(so_tracker, 'anl4', 5)

th_alpha_list = []

for expr, decay in so_layer:
    for alpha in trade_when_factory("trade_when", expr, "USA"):
        th_alpha_list.append((alpha, decay))

random.seed(42)
random.shuffle(th_alpha_list)        

th_pools = load_task_pool_single(th_alpha_list, 3)

# 自动处理断点，状态键名为 "day3"
single_simulate(th_pools, 'SUBINDUSTRY', 'USA', 'TOP3000', "day3")