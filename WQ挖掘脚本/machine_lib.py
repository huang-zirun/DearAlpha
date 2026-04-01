import requests
from os import environ
from time import sleep
import time
import json
import os
import pandas as pd
import random
import pickle
from itertools import product
from itertools import combinations
from collections import defaultdict
from urllib.parse import urljoin
from datetime import datetime

# --- Configuration & Constants ---

CRED_FILE = "credentials.json"
PROG_FILE = "progress.json"

basic_ops = ["reverse", "inverse", "rank", "zscore", "quantile", "normalize"]
ts_ops = ["ts_delta", "ts_sum", "ts_product", "ts_std_dev", "ts_mean", 
          "ts_arg_min", "ts_arg_max", "ts_scale", "normalize", "zscore"]
ops_set = basic_ops + ts_ops 

# --- State Management ---

class ManageState:
    @staticmethod
    def load_state():
        if not os.path.exists(PROG_FILE):
            return {}
        try:
            with open(PROG_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}

    @staticmethod
    def save_state(state):
        with open(PROG_FILE, 'w') as f:
            json.dump(state, f, indent=4)

    @staticmethod
    def get_progress(stage):
        state = ManageState.load_state()
        # Returns default dict if stage not found
        return state.get(stage, {"index": 0, "date": None})

    @staticmethod
    def update_progress(stage, index=None, date=None):
        state = ManageState.load_state()
        if stage not in state:
            state[stage] = {"index": 0, "date": None}
        
        if index is not None:
            state[stage]["index"] = index
        if date is not None:
            state[stage]["date"] = date
            
        ManageState.save_state(state)

# --- Authentication ---

def login():
    # 检查凭证文件是否存在
    if not os.path.exists(CRED_FILE):
        raise FileNotFoundError(f"Error: {CRED_FILE} not found.")
        
    with open(CRED_FILE, 'r') as f:
        creds = json.load(f)

    # 死循环重试，直到登录成功才返回
    while True:
        try:
            s = requests.Session()
            s.auth = (creds['username'], creds['password'])
            response = s.post('https://api.worldquantbrain.com/authentication')
            
            if response.status_code == 201:
                # 只有这里才会跳出循环
                print(f"Login successful (User ID: {response.json()['user']['id']})")
                return s
            
            elif response.status_code == 429:
                # 遇到限流，强制休眠 60 秒
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"[Login] Rate Limit (429). Sleeping for {retry_after}s...")
                time.sleep(retry_after)
                
            else:
                # 其他错误（如密码错误或服务器挂了），休眠 10 秒
                print(f"[Login] Failed: {response.content}. Retrying in 10s...")
                time.sleep(10)
                
        except Exception as e:
            print(f"[Login] Connection Error: {e}. Retrying in 10s...")
            time.sleep(10)

# --- Simulation Core ---

def load_task_pool_single(alpha_list, limit_of_single_simulations):
    '''
    Input:
        alpha_list : list of (alpha, decay) tuples
        limit_of_single_simulations : number of concurrent single simulations
    Output:
        pool : [ alpha_num/3 * [3 * (alpha, decay)] ] 
    '''
    pool = [alpha_list[i:i + limit_of_single_simulations] for i in range(0, len(alpha_list), limit_of_single_simulations)]
    return pool

def single_simulate(alpha_pool, neut, region, universe, stage_name):
    s = login()
    
    # Load progress
    progress = ManageState.get_progress(stage_name)
    start_index = progress["index"]
    
    # If starting fresh, record the date for subsequent stages
    if start_index == 0:
        current_date = datetime.now().strftime("%m-%d")
        ManageState.update_progress(stage_name, date=current_date)
        print(f"[{stage_name}] Starting new run. Date recorded: {current_date}")
    else:
        print(f"[{stage_name}] Resuming from task index: {start_index}")

    for x, task in enumerate(alpha_pool):
        if x < start_index: 
            continue
            
        progress_urls = []
        for y, (alpha, decay) in enumerate(task):
            simulation_data = {
                'type': 'REGULAR',
                'settings': {
                    'instrumentType': 'EQUITY',
                    'region': region, 
                    'universe': universe, 
                    'delay': 1,
                    'decay': decay, 
                    'neutralization': neut,
                    'truncation': 0.08,
                    'pasteurization': 'ON',
                    'testPeriod': 'P0Y',
                    'unitHandling': 'VERIFY',
                    'nanHandling': 'ON',
                    'language': 'FASTEXPR',
                    'visualization': False,
                },
            'regular': alpha}

            try:
                simulation_response = s.post('https://api.worldquantbrain.com/simulations', json=simulation_data)
                if 'Location' in simulation_response.headers:
                    simulation_progress_url = simulation_response.headers['Location']
                    progress_urls.append(simulation_progress_url)
                else:
                    print("Location key error: %s" % simulation_response.content)
                    sleep(60) # Short wait before retry logic might be needed, or just skip
            except Exception as e:
                print(f"Post error: {e}")
                sleep(600)
                s = login()

        print("task %d post done"%(x))

        for j, progress_url in enumerate(progress_urls):
            try:
                while True:
                    simulation_progress = s.get(progress_url)
                    retry_after = float(simulation_progress.headers.get("Retry-After", 0))
                    if retry_after == 0:
                        break
                    sleep(retry_after)

                status = simulation_progress.json().get("status", 0)
                if status != "COMPLETE" and status != "WARNING":
                    print("Not complete : %s" % progress_url)

            except KeyError:
                print("look into: %s" % progress_url)
            except Exception as e:
                print(f"Error checking progress: {e}")

        # Save progress after each task completion
        ManageState.update_progress(stage_name, index=x + 1)
        print("task %d simulate done"%(x))
    
    print(f"[{stage_name}] All simulations done.")

# --- Alpha Retrieval & Processing ---

def get_alphas(start_date, end_date, sharpe_th, fitness_th, region, alpha_num, usage):
    s = login()
    output = []
    count = 0
    
    # Determine year for date construction
    current_year = datetime.now().year
    
    for i in range(0, alpha_num, 100):
        print(f"Fetching alphas offset {i}...")
        
        base_url = "https://api.worldquantbrain.com/users/self/alphas?limit=100&offset=%d"%(i) \
                + "&status=UNSUBMITTED%1FIS_FAIL" \
                + f"&dateCreated%3E={current_year}-{start_date}T00:00:00-04:00" \
                + f"&dateCreated%3C{current_year}-{end_date}T00:00:00-04:00" \
                + "&settings.region=" + region + "&hidden=false&type!=SUPER"

        url_e = base_url + "&is.fitness%3E" + str(fitness_th) + "&is.sharpe%3E" + str(sharpe_th) + "&order=-is.sharpe"
        url_c = base_url + "&is.fitness%3C-" + str(fitness_th) + "&is.sharpe%3C-" + str(sharpe_th) + "&order=is.sharpe"
        
        urls = [url_e]
        if usage != "submit":
            urls.append(url_c)
            
        for url in urls:
            try:
                response = s.get(url)
                alpha_list = response.json().get("results", [])
                
                for j in range(len(alpha_list)):
                    alpha_id = alpha_list[j]["id"]
                    sharpe = alpha_list[j]["is"]["sharpe"]
                    fitness = alpha_list[j]["is"]["fitness"]
                    turnover = alpha_list[j]["is"]["turnover"]
                    margin = alpha_list[j]["is"]["margin"]
                    longCount = alpha_list[j]["is"]["longCount"]
                    shortCount = alpha_list[j]["is"]["shortCount"]
                    decay = alpha_list[j]["settings"]["decay"]
                    exp = alpha_list[j]['regular']['code']
                    dateCreated = alpha_list[j]["dateCreated"]
                    
                    if (longCount + shortCount) > 100:
                        count += 1
                        if sharpe < -sharpe_th:
                            exp = "-%s"%exp
                        
                        rec = [alpha_id, exp, sharpe, turnover, fitness, margin, dateCreated, decay]
                        print(rec)
                        
                        # Adjust decay based on turnover (original logic)
                        if turnover > 0.7:
                            rec.append(decay*4)
                        elif turnover > 0.6:
                            rec.append(decay*3+3)
                        elif turnover > 0.5:
                            rec.append(decay*3)
                        elif turnover > 0.4:
                            rec.append(decay*2)
                        elif turnover > 0.35:
                            rec.append(decay+4)
                        elif turnover > 0.3:
                            rec.append(decay+2)
                        else:
                            rec.append(decay) # Fallback if turnover is low but passes filter
                            
                        output.append(rec)
            except Exception as e:
                print(f"{i} error or re-login needed: {e}")
                s = login()

    print("count: %d"%count)
    return output

def prune(next_alpha_recs, prefix, keep_num):
    output = []
    num_dict = defaultdict(int)
    for rec in next_alpha_recs:
        exp = rec[1]
        try:
            # Simple heuristic to extract field name based on prefix
            if prefix in exp:
                field = exp.split(prefix)[-1].split(",")[0]
            else:
                field = exp # Fallback if prefix structure doesn't match
                
            sharpe = rec[2]
            if sharpe < 0:
                field = "-%s"%field
            
            if num_dict[field] < keep_num:
                num_dict[field] += 1
                decay = rec[-1]
                output.append([exp, decay])
        except:
            continue
    return output

# --- Data Processing Helpers ---

def get_datasets(s, instrument_type='EQUITY', region='USA', delay=1, universe='TOP3000'):
    url = "https://api.worldquantbrain.com/data-sets?" +\
        f"instrumentType={instrument_type}&region={region}&delay={str(delay)}&universe={universe}"
    result = s.get(url)
    datasets_df = pd.DataFrame(result.json()['results'])
    return datasets_df

def get_datafields(s, instrument_type='EQUITY', region='USA', delay=1, universe='TOP3000', dataset_id='', search=''):
    if len(search) == 0:
        url_template = "https://api.worldquantbrain.com/data-fields?" +\
            f"instrumentType={instrument_type}" +\
            f"&region={region}&delay={str(delay)}&universe={universe}&dataset.id={dataset_id}&limit=50" +\
            "&offset={x}"
        try:
            count = s.get(url_template.format(x=0)).json()['count'] 
        except:
            count = 100
    else:
        url_template = "https://api.worldquantbrain.com/data-fields?" +\
            f"&instrumentType={instrument_type}" +\
            f"&region={region}&delay={str(delay)}&universe={universe}&limit=50" +\
            f"&search={search}" +\
            "&offset={x}"
        count = 100
    
    datafields_list = []
    for x in range(0, count, 50):
        while True:
            try:
                datafields = s.get(url_template.format(x=x))
                if datafields.status_code == 200:
                    datafields_list.append(datafields.json().get('results', []))
                    break
                elif datafields.status_code == 429:
                    retry_after = int(datafields.headers.get("Retry-After", 60))
                    print(f"Rate limited at offset {x}. Retrying in {retry_after}s...")
                    time.sleep(retry_after)
                else:
                    print(f"Error fetching offset {x}: {datafields.status_code}. Retrying in 10s...")
                    time.sleep(10)
            except Exception as e:
                print(f"Connection error at offset {x}: {e}. Retrying in 10s...")
                time.sleep(10)
 
    datafields_list_flat = [item for sublist in datafields_list for item in sublist]
    datafields_df = pd.DataFrame(datafields_list_flat)
    return datafields_df

def get_vec_fields(fields):
    vec_ops = ["vec_avg", "vec_sum"]
    vec_fields = []
    for field in fields:
        for vec_op in vec_ops:
            if vec_op == "vec_choose":
                vec_fields.append("%s(%s, nth=-1)"%(vec_op, field))
                vec_fields.append("%s(%s, nth=0)"%(vec_op, field))
            else:
                vec_fields.append("%s(%s)"%(vec_op, field))
    return vec_fields

def process_datafields(df):
    if df.empty or 'type' not in df.columns:
        print("Warning: Dataframe is empty or missing 'type' column.")
        return []
    datafields = []
    datafields += df[df['type'] == "MATRIX"]["id"].tolist()
    datafields += get_vec_fields(df[df['type'] == "VECTOR"]["id"].tolist())
    return datafields

def process_datafields1(df):
    datafields = []
    datafields += df[df['type'] == "MATRIX"]["id"].tolist()
    datafields += get_vec_fields(df[df['type'] == "VECTOR"]["id"].tolist())
    return ["winsorize(ts_backfill(%s, 120), std=4)"%field for field in datafields]

def normalize(df):
    add = []
    for i in df:
        add.append("normalize(%s, useStd = false, limit = 0.0)"%i)
    for j in df:
        add.append("ts_delta(ts_delta(%s, 20),20)"%j)
    for k in df:
        add.append("group_rank(%s, subindustry)-group_rank(%s, market)"%(k, k))
    for l in df:
        add.append(" group rank(%s, market)-group_rank(group_mean(%s, subindustry), market)"%(l, l))
    return add

def add():
    datafields = [
        "ts_arg_min(fnd6_cptmfmq_opepsq/fnd6_txw, 120)",
        # ... (Previous list content omitted for brevity, but logic structure retained) ...
        # If specific list content is strictly required, insert the full list from original file here.
        # Assuming the function logic is what matters:
        "ts_sum(fnd6_cibegni/fnd6_newa2v1300_ni, 240)"
    ]
    added = []
    for i in datafields:
        for j in datafields:
            if i != j:
                added.append("winsorize((%s+%s),std=4)"%(i, j))
    return added

def fnd6_fields(df, df1=None):
    # Handles both signatures found in original file (single list or two lists)
    vec_fields = []
    if df1 is None:
        # Signature 2 behavior: field/field
        for field in df:
            for vec_op in df:
                if vec_op != field:
                    vec_fields.append("%s/%s"%(vec_op, field))
    else:
        # Signature 1 behavior: vec_op/field
        for field in df:
            for vec_op in df1:
                if vec_op != field:
                    vec_fields.append("ts_backfill( winsorize(%s/%s,std=4),120)"%(vec_op, field))
    return vec_fields

def model77(df):
    doubao_fields_1 = [
        "mdl77_2deepvaluefactor_pedwf - mdl77_2deepvaluefactor_estep",
        "(mdl77_2400_yen + mdl77_2deepvaluefactor_pfcfmtt) / 2",
        "mdl77_2earningmomentumfactor400_gspea2y - mdl77_2earningmomentumfactor400_gspea1y",
        "mdl77_2gdna_pctchgocf * mdl77_2gdna_pctchgcf",
        "mdl77_2gdna_roic / mdl77_2gdna_susgrowth",
        "mdl77_2400_impvol - mdl77_2400_rmi",
        "mdl77_2gdna_ttmaccu / mdl77_2gdna_ocfast",
        "mdl77_2gdna_debtcf * mdl77_2gdna_cfleverage",
        "mdl77_2400_chg12msip / mdl77_2400_chgshare",
        "mdl77_2gdna_indrelrtn5d_ * mdl77_2gdna_visiratio",
        "mdl77_2earningmomentumfactor400_numrevq1 - mdl77_2earningmomentumfactor400_numrevy1",
        "mdl77_2gdna_fixastto / mdl77_2gdna_astto",
        "(mdl77_2gdna_ocfmargin - mdl77_2gdna_mpn) * 100",
        "mdl77_2gdna_salerec / mdl77_2gdna_pca",
        "mdl77_2gdna_rel5yep / mdl77_2gdna_rel5yfcfp",
        "mdl77_2gdna_tobinq * mdl77_2gdna_pvan",
        "mdl77_2gdna_ebitdaev - mdl77_2gdna_vefcfmtt"
    ]
    return doubao_fields_1

# --- Factories ---

def ts_factory(op, field):
    output = []
    days = [5, 22, 66, 120, 240]
    for day in days:
        alpha = "%s(%s, %d)"%(op, field, day)
        output.append(alpha)
    return output

def ts_comp_factory(op, field, factor, paras):
    output = []
    l1, l2 = [5, 22, 66, 240], paras
    comb = list(product(l1, l2))
    for day, para in comb:
        if type(para) == float:
            alpha = "%s(%s, %d, %s=%.1f)"%(op, field, day, factor, para)
        elif type(para) == int:
            alpha = "%s(%s, %d, %s=%d)"%(op, field, day, factor, para)
        output.append(alpha)
    return output

def vector_factory(op, field):
    output = []
    vectors = ["cap"]
    for vector in vectors:
        alpha = "%s(%s, %s)"%(op, field, vector)
        output.append(alpha)
    return output

def twin_field_factory(op, field, fields):
    output = []
    days = [5, 22, 66, 240]
    outset = list(set(fields) - set([field]))
    for day in days:
        for counterpart in outset:
            alpha = "%s(%s, %s, %d)"%(op, field, counterpart, day)
            output.append(alpha)
    return output

def first_order_factory(fields, ops_set):
    alpha_set = []
    for field in fields:
        alpha_set.append(field)
        for op in ops_set:
            if op == "ts_percentage":
                alpha_set += ts_comp_factory(op, field, "percentage", [0.5])
            elif op == "ts_decay_exp_window":
                alpha_set += ts_comp_factory(op, field, "factor", [0.5])
            elif op == "ts_moment":
                alpha_set += ts_comp_factory(op, field, "k", [2, 3, 4])
            elif op == "ts_entropy":
                alpha_set += ts_comp_factory(op, field, "buckets", [10])
            elif op.startswith("ts_") or op == "inst_tvr":
                alpha_set += ts_factory(op, field)
            elif op.startswith("vector"):
                alpha_set += vector_factory(op, field)
            elif op == "signed_power":
                alpha = "%s(%s, 2)"%(op, field)
                alpha_set.append(alpha)
            elif op == "normalize":
                alpha = "%s(%s, useStd = false, limit = 0.0)"%(op, field)
                alpha_set.append(alpha)
            else:
                alpha = "%s(%s)"%(op, field)
                alpha_set.append(alpha)
    return alpha_set

def get_group_second_order_factory(first_order, group_ops, region):
    second_order = []
    for fo in first_order:
        for group_op in group_ops:
            second_order += group_factory(group_op, fo, region)
    return second_order

def group_factory(op, field, region):
    output = []
    vectors = ["cap"] 
    usa_group_13 = ['pv13_h_min2_3000_sector','pv13_r2_min20_3000_sector','pv13_r2_min2_3000_sector',
                    'pv13_r2_min2_3000_sector', 'pv13_h_min2_focused_pureplay_3000_sector']
    
    cap_group = "bucket(rank(cap), range='0.1, 1, 0.1')"
    asset_group = "bucket(rank(assets),range='0.1, 1, 0.1')"
    sector_cap_group = "bucket(group_rank(cap, sector),range='0.1, 1, 0.1')"
    sector_asset_group = "bucket(group_rank(assets, sector),range='0.1, 1, 0.1')"
    vol_group = "bucket(rank(ts_std_dev(returns,20)),range = '0.1, 1, 0.1')"
    liquidity_group = "bucket(rank(close*volume),range = '0.1, 1, 0.1')"

    groups = ["market","sector", "industry", "subindustry",
              cap_group, asset_group, sector_cap_group, sector_asset_group, vol_group, liquidity_group]
    groups += usa_group_13
        
    for group in groups:
        if op.startswith("group_vector"):
            for vector in vectors:
                alpha = "%s(%s,%s,densify(%s))"%(op, field, vector, group)
                output.append(alpha)
        elif op.startswith("group_percentage"):
            alpha = "%s(%s,densify(%s),percentage=0.5)"%(op, field, group)
            output.append(alpha)
        else:
            alpha = "%s(%s,densify(%s))"%(op, field, group)
            output.append(alpha)
    return output

def trade_when_factory(op, field, region):
    output = []
    open_events = ["pcr_oi_270<1","ts_arg_max(volume, 5) == 0", "ts_corr(close, volume, 20) < 0",
                   "ts_corr(close, volume, 5) < 0", "ts_mean(volume,10)>ts_mean(volume,60)",
                   "group_rank(ts_std_dev(returns,60), sector) > 0.7", "ts_zscore(returns,60) > 2",
                   "ts_arg_min(volume, 5) > 3",
                   "ts_std_dev(returns, 5) > ts_std_dev(returns, 20)",
                   "ts_arg_max(close, 5) == 0", "ts_arg_max(close, 20) == 0",
                   "ts_corr(close, volume, 5) > 0", "ts_corr(close, volume, 5) > 0.3", "ts_corr(close, volume, 5) > 0.5",
                   "ts_corr(close, volume, 20) > 0", "ts_corr(close, volume, 20) > 0.3", "ts_corr(close, volume, 20) > 0.5",
                   "ts_regression(returns, %s, 5, lag = 0, rettype = 2) > 0"%field,
                   "ts_regression(returns, %s, 20, lag = 0, rettype = 2) > 0"%field,
                   "ts_regression(returns, ts_step(20), 20, lag = 0, rettype = 2) > 0",
                   "ts_regression(returns, ts_step(5), 5, lag = 0, rettype = 2) > 0"]

    exit_events = ["abs(returns) > 0.1", "-1"]
    
    # Keeping original event lists though trade_when typically uses open_events
    usa_events = ["rank(rp_css_business) > 0.8", "pcr_oi_270 < 1"]

    for oe in open_events:
        for ee in exit_events:
            alpha = "%s(%s, %s, %s)"%(op, oe, field, ee)
            output.append(alpha)
    return output

# --- Submission & Analysis ---

def check_submission(alpha_bag, gold_bag, start):
    depot = []
    s = login()
    for idx, g in enumerate(alpha_bag):
        if idx < start:
            continue
        if idx % 5 == 0:
            print(idx)
        if idx % 200 == 0:
            s = login()
        pc = get_check_submission(s, g)
        if pc == "sleep":
            sleep(100)
            s = login()
            alpha_bag.append(g)
        elif pc != pc:
            print("check self-corrlation error")
            sleep(100)
            alpha_bag.append(g)
        elif pc == "fail":
            continue
        elif pc == "error":
            depot.append(g)
        else:
            print(g)
            gold_bag.append((g, pc))
    print(depot)
    return gold_bag

def get_check_submission(s, alpha_id):
    while True:
        try:
            result = s.get("https://api.worldquantbrain.com/alphas/" + alpha_id + "/check")
            if "retry-after" in result.headers:
                time.sleep(float(result.headers["Retry-After"]))
            else:
                break
        except:
            time.sleep(10)
    try:
        if result.json().get("is", 0) == 0:
            print("logged out")
            return "sleep"
        checks_df = pd.DataFrame(result.json()["is"]["checks"])
        pc = checks_df[checks_df.name == "SELF_CORRELATION"]["value"].values[0]
        if not any(checks_df["result"] == "FAIL"):
            return pc
        else:
            return "fail"
    except:
        print("catch: %s"%(alpha_id))
        return "error"

def view_alphas(gold_bag):
    s = login()
    sharp_list = []
    for gold, pc in gold_bag:
        triple = locate_alpha(s, gold)
        info = [triple[0], triple[2], triple[3], triple[4], triple[5], triple[6], triple[1]]
        info.append(pc)
        sharp_list.append(info)
    sharp_list.sort(reverse=True, key = lambda x : x[1])
    for i in sharp_list:
        print(i)
 
def locate_alpha(s, alpha_id):
    while True:
        alpha = s.get("https://api.worldquantbrain.com/alphas/" + alpha_id)
        if "retry-after" in alpha.headers:
            time.sleep(float(alpha.headers["Retry-After"]))
        else:
            break
    string = alpha.content.decode('utf-8')
    metrics = json.loads(string)
    
    dateCreated = metrics["dateCreated"]
    sharpe = metrics["is"]["sharpe"]
    fitness = metrics["is"]["fitness"]
    turnover = metrics["is"]["turnover"]
    margin = metrics["is"]["margin"]
    decay = metrics["settings"]["decay"]
    exp = metrics['regular']['code']
    
    triple = [alpha_id, exp, sharpe, turnover, fitness, margin, dateCreated, decay]
    return triple

def set_alpha_properties(s, alpha_id, name=None, color=None, selection_desc="None", combo_desc="None", tags=["ace_tag"]):
    params = {
        "color": color,
        "name": name,
        "tags": tags,
        "category": None,
        "regular": {"description": None},
        "combo": {"description": combo_desc},
        "selection": {"description": selection_desc},
    }
    response = s.patch("https://api.worldquantbrain.com/alphas/" + alpha_id, json=params)

# --- Multi-Simulation / Consultant ---

def generate_sim_data(alpha_list, region, uni, neut):
    sim_data_list = []
    for alpha, decay in alpha_list:
        simulation_data = {
            'type': 'REGULAR',
            'settings': {
                'instrumentType': 'EQUITY',
                'region': region,
                'universe': uni,
                'delay': 1,
                'decay': decay,
                'neutralization': neut,
                'truncation': 0.08,
                'pasteurization': 'ON',
                'testPeriod': 'P2Y',
                'unitHandling': 'VERIFY',
                'nanHandling': 'ON',
                'language': 'FASTEXPR',
                'visualization': False,
            },
            'regular': alpha}
        sim_data_list.append(simulation_data)
    return sim_data_list

def load_task_pool(alpha_list, limit_of_children_simulations, limit_of_multi_simulations):
    tasks = [alpha_list[i:i + limit_of_children_simulations] for i in range(0, len(alpha_list), limit_of_children_simulations)]
    pools = [tasks[i:i + limit_of_multi_simulations] for i in range(0, len(tasks), limit_of_multi_simulations)]
    return pools

def multi_simulate(alpha_pools, neut, region, universe, start):
    s = login()
    for x, pool in enumerate(alpha_pools):
        if x < start: continue
        progress_urls = []
        for y, task in enumerate(pool):
            sim_data_list = generate_sim_data(task, region, universe, neut)
            try:
                simulation_response = s.post('https://api.worldquantbrain.com/simulations', json=sim_data_list)
                if 'Location' in simulation_response.headers:
                    progress_urls.append(simulation_response.headers['Location'])
                else:
                    sleep(600)
                    s = login()
            except:
                sleep(600)
                s = login()

        print("pool %d task %d post done"%(x,y))

        for j, progress in enumerate(progress_urls):
            try:
                while True:
                    simulation_progress = s.get(progress)
                    if simulation_progress.headers.get("Retry-After", 0) == 0:
                        break
                    sleep(float(simulation_progress.headers["Retry-After"]))

                status = simulation_progress.json().get("status", 0)
                if status != "COMPLETE":
                    print("Not complete : %s"%(progress))
            except:
                print("error checking progress")

        print("pool %d task %d simulate done"%(x, j))
    print("Simulate done")