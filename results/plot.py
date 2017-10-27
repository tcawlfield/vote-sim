#!/usr/bin/python3

import sys
import re
import csv
import numpy
import matplotlib.pyplot as plt

in_csv = sys.argv[1]
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

ys = list(range(1, len(data_cols)))

results_nz = dict()
for vm in data_cols[1:]:
    mean = numpy.mean(results[vm])
    nonzero = [x for x in results[vm] if x > 0.0]
    mean_nz = numpy.mean(nonzero)
    results_nz[vm] = nonzero
    frac_nz = float(len(nonzero)) / float(len(results[vm]))
    print("{:16s}: avg {:.3f}  {:.3f}% non-zero  {:.3f} mean-non-zero".format(vm, mean, frac_nz*100, mean_nz))

# fig, ax = plt.subplots()
# ax.errorbar(xs, ys, xerr=x_errs, fmt='o')
# ax.set_yticks(ys)
# ax.set_yticklabels(data_cols)

fig, (mva, rega) = plt.subplots(2)
mva.hist(results['SPlMargin'], histtype='step')
mva.set_title('Margin of victory, strategic plurality')

bpdata = []
bplabels = []
voting_methods = []
for vm in data_cols:
    m = re.match(r'^(.+)Regret', vm)
    if m:
        bpdata.append(results[vm])
        bplabels.append(m.group(1))
        voting_methods.append(vm)
rega.boxplot(bpdata, vert=False, labels=bplabels, whis=[1, 99], sym='')
rega.set_title('Regrets')
fig.tight_layout()

fig, vm_axs = plt.subplots(len(voting_methods))
for i, vm in enumerate(voting_methods):
    vm_axs[i].hist(results_nz[vm], histtype='step')
    vm_axs[i].set_title(bplabels[i])
fig.tight_layout()

plt.show(block=False)
s = input('Press Enter to exit\n')
