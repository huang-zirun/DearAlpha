from machine_lib import * 
s = login()

# Setup Data
df = get_datafields(s, dataset_id='analyst4', region='USA', universe='TOP3000', delay=1)
pc_fields = process_datafields(df)
first_order = first_order_factory(pc_fields, ts_ops)

init_decay = 6
fo_alpha_list = [(alpha, init_decay) for alpha in first_order]

random.seed(42)
random.shuffle(fo_alpha_list)

fo_pools = load_task_pool_single(fo_alpha_list, 3)

# 自动处理断点和日期记录，状态键名为 "day1"
single_simulate(fo_pools, "SUBINDUSTRY", "USA", "TOP3000", "day1")