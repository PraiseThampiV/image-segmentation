#adapted from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataset.py
import itertools
from torch.utils.data.dataset import IterableDataset
class ChainDataset(IterableDataset):
    r"""modified chainDataset of pytorch Dataset for chainning multiple :class:`IterableDataset` s.

    This class is useful to assemble different existing dataset streams. The
    chainning operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.

    Arguments:
        datasets (iterable of IterableDataset): datasets to be chained together

    length gives only the total count of elements in each dataset without including count when cycled
    """
    def __init__(self, datasets, cnt_cycle):
        super(ChainDataset, self).__init__()
        self.datasets = datasets
        self.cnt_cycle = cnt_cycle
        self.iter_datasets = itertools.chain.from_iterable(itertools.repeat(self.datasets, self.cnt_cycle))
        # above line produces an iter having patchsets (as its items) with length of 10, 13, 8, 4, etc
        # below lone, first cnt_cycle times an iter, then iter combining all iters in selfdatasets
        self.iter_datasets=itertools.tee(itertools.chain(self.datasets), self.cnt_cycle)


    def __iter__(self):     
        for d in self.datasets:
            #assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            for x in d:
                yield x

    def __len__(self):
        total = 0
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            total += len(d)
        return total