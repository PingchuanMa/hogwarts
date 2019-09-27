import time
import statistics
from collections import OrderedDict, deque
from contextlib import contextmanager
from collections.abc import Iterable


class RecordManager:
    def __init__(self):
        self.groups = OrderedDict()
        self.stats = OrderedDict()
        self.records = OrderedDict()
        self.ticks = dict()

    def record_value(self, key, value, window_size, summary_mode, group='default'):
        if window_size != 'inf' and (not isinstance(window_size, int) or window_size <= 0):
            raise ValueError('window_size should be positive integer or string "inf"')
        if summary_mode not in ['mean', 'sum', 'max', 'min', 'stat']:
            raise ValueError('summary_mode should be one of mean / sum / max / min / stat')
        Record = {
            'mean': MeanRecord,
            'sum': SumRecord,
            'max': MaxRecord,
            'min': MinRecord,
            'stat': StatRecord,
        }[summary_mode]
        groups = self.stats if summary_mode == 'stat' else self.groups
        if group not in groups:
            groups[group] = OrderedDict()
        if key not in groups[group]:
            groups[group][key] = Record(window_size)
        record = groups[group][key]
        if not isinstance(record, Record):
            raise TypeError('renew record {} in group {} with different summary_mode'
                            'is not allowed'.format(repr(key), repr(group)))
        if record.window_size != window_size:
            raise ValueError('renew record {} in group {} with different window_size'
                             'is not allowed'.format(repr(key), repr(group)))
        record.record_value(value)

    def record_tick(self, key):
        self.ticks[key] = time.time()

    def record_tock(self, key, window_size, summary_mode, group='default'):
        if key not in self.ticks:
            raise Exception('key {} has not been ticked'.format(repr(key)))
        value = time.time() - self.ticks[key]
        self.record_value(key, value, window_size, summary_mode, group)

    @contextmanager
    def record_time(self, key, window_size, summary_mode, group='default'):
        self.record_tick(key)
        yield
        self.record_tock(key, window_size, summary_mode, group)

    def summary(self, key, group='default', stat=False):
        groups = self.stats if stat else self.groups
        if group not in groups:
            raise KeyError('group {} not exists'.format(repr(group)))
        if key not in groups[group]:
            raise KeyError('record {} not exists'.format(repr(key)))
        return groups[group][key].summary()

    # =========================
    # dictionary-like interface
    # =========================

    def keys(self, group='default', stat=False):
        groups = self.stats if stat else self.groups
        if group not in groups:
            raise KeyError('group {} not exists'.format(repr(group)))
        return list(groups[group].keys())

    def values(self, group='default', stat=False):
        groups = self.stats if stat else self.groups
        if group not in groups:
            raise KeyError('group {} not exists'.format(repr(group)))
        return [record.summary() for record in groups[group].values()]

    def items(self, group='default', stat=False):
        groups = self.stats if stat else self.groups
        if group not in groups:
            raise KeyError('group {} not exists'.format(repr(group)))
        return [(key, record.summary()) for key, record in groups[group].items()]

    def clear_record(self, key, group='default', stat=False):
        groups = self.stats if stat else self.groups
        if group not in groups:
            raise KeyError('group {} not exists'.format(repr(group)))
        if key not in groups[group]:
            raise KeyError('record {} not exists'.format(repr(key)))
        del groups[group][key]

    def clear_records(self, group='default', stat=False):
        groups = self.stats if stat else self.groups
        if group not in groups:
            raise KeyError('group {} not exists'.format(repr(group)))
        groups[group].clear()


class MeanRecord:
    def __init__(self, window_size):
        self.empty = True
        self.window_size = window_size
        if window_size == 'inf':
            self.window = []
        else:
            self.window = deque(maxlen=window_size)

    def record_value(self, value):
        self.empty = False
        if isinstance(value, Iterable):
            self.window.extend(value)
        else:
            self.window.append(value)

    def summary(self):
        assert not self.empty, 'empty record'
        return sum(self.window) / len(self.window)


class SumRecord:
    def __init__(self, window_size):
        self.empty = True
        self.window_size = window_size
        if window_size == 'inf':
            self.value = 0
        else:
            self.window = deque(maxlen=window_size)

    def record_value(self, value):
        self.empty = False
        if self.window_size == 'inf':
            self.value += value
        else:
            if isinstance(value, Iterable):
                self.window.extend(value)
            else:
                self.window.append(value)

    def summary(self):
        assert not self.empty, 'empty record'
        if self.window_size == 'inf':
            return self.value
        else:
            return sum(self.window)


class _MinMaxRecord:
    def __init__(self, window_size):
        self.window_size = window_size
        self.empty = True
        if window_size != 'inf':
            self.queue = deque(maxlen=window_size)

    def record_value(self, value):
        self.empty = False
        if hasattr(self, 'queue'):
            self.queue.append(value)
        elif hasattr(self, 'value'):
            self.value = self.minmax(self.value, value)
        else:
            self.value = value

    def summary(self):
        assert not self.empty, 'empty record'
        if hasattr(self, 'queue'):
            return self.minmax(self.queue)
        return self.value


class MinRecord(_MinMaxRecord):
    minmax = min


class MaxRecord(_MinMaxRecord):
    minmax = max


class StatRecord:
    def __init__(self, window_size):
        self.empty = True
        self.window_size = window_size
        if window_size == 'inf':
            self.window = []
        else:
            self.window = deque(maxlen=window_size)

    def record_value(self, value):
        self.empty = False
        if isinstance(value, Iterable):
            self.window.extend(value)
        else:
            self.window.append(value)

    def summary(self):
        assert not self.empty, 'empty record'
        mean = statistics.mean(self.window)
        var = statistics.variance(self.window, mean)
        return {
            'mean': mean,
            'var': var,
            'max': max(self.window),
            'min': min(self.window)
        }
