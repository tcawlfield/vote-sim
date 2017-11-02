import numpy
from sklearn.datasets.base import Bunch

from read import Scenario

class SimvoteDataset(Scenario):
    def get_sk_dataset(self, method, as_classification=True):
        ncand = len(self.regrets[0])
        feature_names = []
        cov_feature_indices = []
        for i in range(ncand):
            feature_names.append('regret[{}]'.format(i))
        for ix, row in enumerate(self.cov_mats[0]):
            for iy in range(len(row)):
                name = 'cov[{},{}]'.format(ix, iy)
                feature_names.append(name)
                cov_feature_indices.append((ix, iy))

        data = []
        for r, c in zip(self.regrets, self.cov_mats):
            row = list(r)  # copy
            for ix, iy in cov_feature_indices:
                row.append(c[ix][iy])
            data.append(row)

        target = []
        if as_classification:
            for r in self.method_regrets[method]:
                target.append(0 if r == 0.0 else 1)
        else:
            target = self.method_regrets[method]

        ds = Bunch(
            DESCR=method,
            data=numpy.array(data),
            target=numpy.array(target),
            feature_names=feature_names,
        )
        if as_classification:
            ds['target_names'] = ['best', 'suboptimal']
        return ds
