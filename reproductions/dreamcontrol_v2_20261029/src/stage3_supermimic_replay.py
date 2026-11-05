"""Bounded device replay for real, ordered simulator transitions.

Only actual transitions are stored. The world-model loss masks sequence tails
after an episode termination; terminal next states are captured before reset.
"""
import torch


FIELDS = {
    's': (112, torch.float32), 's1': (112, torch.float32),
    'a': (31, torch.float32), 'a_prev': (31, torch.float32),
    'ref': (109, torch.float32), 'r': (None, torch.float32),
    'term': (None, torch.bool), 'done': (None, torch.bool),
    'id': (None, torch.int64), 'step': (None, torch.int64),
}


class RealReplay:
    def __init__(self, capacity, num_envs, device):
        if capacity < 8 or num_envs < 1:
            raise ValueError('invalid replay dimensions')
        self.capacity, self.num_envs = capacity, num_envs
        self.data = {k: torch.zeros((capacity, num_envs)+(() if n is None else (n,)),
                                   dtype=dtype, device=device)
                     for k, (n, dtype) in FIELDS.items()}
        self.count = self.position = self.total_rows = 0

    @property
    def bytes(self):
        return sum(v.numel()*v.element_size() for v in self.data.values())

    def append(self, row):
        if set(row) != set(self.data):
            raise ValueError('transition fields differ')
        for k, dest in self.data.items():
            source = row[k]
            if source.shape != dest[self.position].shape or source.dtype != dest.dtype:
                raise ValueError('transition shape/dtype differs: '+k)
            if not torch.isfinite(source).all():
                raise ValueError('nonfinite real transition: '+k)
        for k, dest in self.data.items():
            dest[self.position].copy_(row[k])
        self.position = (self.position+1) % self.capacity
        self.count = min(self.count+1, self.capacity)
        self.total_rows += 1

    def sample(self, batch_size, length):
        if batch_size < 1 or not 1 <= length <= self.count:
            raise ValueError('not enough real replay rows')
        device = self.data['s'].device
        # Logical rows are oldest->newest even after the ring wraps.
        begin = self.position if self.count == self.capacity else 0
        starts = torch.randint(self.count-length+1, (batch_size,), device=device)
        rows = (begin+starts[:, None]+torch.arange(length, device=device)) % self.capacity
        envs = torch.randint(self.num_envs, (batch_size,), device=device)
        return {k: value[rows, envs[:, None]] for k, value in self.data.items()}

    def state_dict(self):
        return {'schema': 1, 'capacity': self.capacity, 'num_envs': self.num_envs,
                'count': self.count, 'position': self.position, 'total_rows': self.total_rows,
                'data': self.data}

    def load_state_dict(self, state):
        if (state['schema'], state['capacity'], state['num_envs']) != (1, self.capacity, self.num_envs):
            raise ValueError('restored replay dimensions differ')
        count, pos, total = state['count'], state['position'], state['total_rows']
        if not 0 <= count <= self.capacity or not 0 <= pos < self.capacity or total < count:
            raise ValueError('invalid replay counters')
        if count != min(total, self.capacity) or pos != total % self.capacity:
            raise ValueError('inconsistent replay history')
        if set(state['data']) != set(self.data):
            raise ValueError('restored replay fields differ')
        for k, dest in self.data.items():
            source = state['data'][k]
            if dest.shape != source.shape or dest.dtype != source.dtype or not torch.isfinite(source).all():
                raise ValueError('invalid restored replay: '+k)
            dest.copy_(source)
        self.count, self.position, self.total_rows = count, pos, total
