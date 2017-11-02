import re
import os
import csv

def read_csv(in_csv):
    """
    Reads the given csv file, possibly a .log file with the same root.
    Returns (metadata, data_cols, results, regrets, cov)
    """
    with open(in_csv) as fin:
        hrdr = csv.DictReader(fin)
        metadata = hrdr.__next__()
        fin.readline()
        print(metadata)

        data_rdr = csv.reader(fin)
        data_cols = data_rdr.__next__()
        results = dict()

        for vm in data_cols:
            results[vm] = []
        for row in data_rdr:
            for i, col in enumerate(row):
                if i < len(data_cols):
                    results[data_cols[i]].append(float(col))
                else:
                    print("Too many values in a row")

    regrets = []
    cov = []
    (base, extn) = os.path.splitext(in_csv)
    logfile = base + '.log'
    if os.path.exists(logfile):
        regret_re = re.compile(r'^Regrets: (\[[\d\.\-, ]+\])$')
        cov_re = re.compile(r'^ \[(\d+)\] +(\-?\d.*)$')
        with open(logfile) as f:
            cov_mat = []
            for line in f:
                line = line.rstrip()
                m = regret_re.search(line)
                if m:
                    latest_regrets = eval(m.group(1))
                    regrets.append(latest_regrets)
                    continue
                m = cov_re.search(line)
                if m:
                    cov_mat.append([float(x) for x in m.group(2).split()])
                    # print("got cov_mat row:", cov_mat)
                    row_num = int(m.group(1))
                    if row_num == len(latest_regrets) - 1:
                        cov.append(cov_mat)
                        cov_mat = []

    return (metadata, data_cols, results, regrets, cov)

class Scenario(object):
    def __init__(self):
        self.metadata = {}
        self.methods = []
        self.method_regrets = {}
        self.pl_margins = []
        self.regrets = []
        self.cov_mats = []
        self.ncand = 0

    @classmethod
    def read(cls, csvfile):
        (metadata, data_cols, results, regrets, cov) = read_csv(csvfile)
        s = cls()
        s.metadata = metadata
        for colname in data_cols:
            m = re.match(r'^(.+)Regret', colname)
            if m:
                vm = m.group(1)
                s.methods.append(vm)
                s.method_regrets[vm] = results[colname]
        s.pl_margins = results['SPlMargin']
        s.regrets = regrets
        s.cov_mats = cov
        s.ncand = len(regrets[0])
        return s
