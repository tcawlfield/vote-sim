from tempfile import TemporaryDirectory
from pathlib import Path
import subprocess
import shutil
import copy
from collections import defaultdict

import tomlkit
import awkward as ak
import pandas as pd
from IPython.display import display

MODULE_PATH = Path(__file__).parent
VOTING = MODULE_PATH / '..' / 'target' / 'release' / 'voting'


def do_run(config, trials) -> ak.Array:
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        config_file = temp_dir / 'config.toml'
        with open(config_file, 'w') as fout:
            tomlkit.dump(config, fout)
        pq_file = temp_dir / 'results.parquet'
        cp = subprocess.run([
            VOTING,
            '-c', config_file,
            '-o', pq_file,
            '-t', str(trials),
        ], check=False, capture_output=True)
        last_config_file = '/tmp/voting_last_config.toml'
        shutil.copyfile(config_file, last_config_file);
        if cp.returncode != 0:
            print(cp.stderr)
            raise ValueError(f"voting process failed -- see perhaps {last_config_file}")
        results = ak.from_parquet(pq_file)
    return results


def get_fields(results):
    return list(filter(lambda fn: fn.startswith('m:'), results.fields))    


def config_series(base_config, parameter, pvalues):
    """
    Generates a sequence of configs based on the base config, with a dotted parameter
    taking on a given sequence of values.
    Returns a generator that yields tuples of (param_name, value, config)
    """
    config = copy.deepcopy(base_config)
    parameters = [int(p) if p.isnumeric() else p for p in parameter.split('.')]
    last_parameter = parameters.pop()
    for val in pvalues:
        conf_item = config
        for p in parameters:  # drill down into config
            conf_item = conf_item[p]
        assert last_parameter in conf_item, f"No such parameter {last_parameter}"
        conf_item[last_parameter] = val
        param_name = parameters[-1] if parameters else last_parameter
        yield (param_name, val, config)


def run_experiment(config_seq, trials=2000):
    iexp = 0
    table_data = defaultdict(list)
    for config_tuple in config_seq:
        if isinstance(config_tuple, tuple):
            (pname, pval, config) = config_tuple
        else:
            (pname, pval, config) = ('config_num', iexp, config_tuple)
        iexp += 1  # Only for the case that config_seq is a list of configs.

        results = do_run(config, trials)
        fields = get_fields(results)
        table_data[pname].append(pval)
        for f in fields:
            mean_regret = ak.mean(results[f]['regret'])
            percent_ideal = ak.count(results[f]['regret'][results[f]['regret'] == 0]) / trials * 100.
            table_data[f'{f[2:]}_mR'].append(mean_regret)
            table_data[f'{f[2:]}_pi'].append(percent_ideal)
    # print(repr(table_data))
    return pd.DataFrame(table_data)
