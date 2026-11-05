"""Real-simulator interaction accounting for the scheduled SuperMimic adapter.

This is preparation only. The future collector must call record_after_step()
after actual simulator steps; imagined world-model transitions never count.
Policy, optimizer, replay and native UTD accumulator restoration are separate.
"""
from dataclasses import dataclass


def natural(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        raise ValueError(name+' must be '+('positive' if positive else 'nonnegative')+' integer')
    return value


@dataclass
class RealInteractionBudget:
    num_envs: int
    macro_rounds: int
    steps_per_round: int
    vector_steps: int = 0
    warmup_vector_steps: int = 0

    def __post_init__(self):
        for name in ['num_envs', 'macro_rounds', 'steps_per_round']:
            natural(getattr(self, name), name, positive=True)
        natural(self.vector_steps, 'vector_steps')
        natural(self.warmup_vector_steps, 'warmup_vector_steps')
        if self.vector_steps > self.target_vector_steps or self.warmup_vector_steps > self.vector_steps:
            raise ValueError('invalid restored interaction counts')

    @property
    def target_vector_steps(self):
        return self.macro_rounds*self.steps_per_round

    @property
    def remaining_vector_steps(self):
        return self.target_vector_steps-self.vector_steps

    @property
    def real_transitions(self):
        return self.num_envs*self.vector_steps

    def cap_collection(self, requested_steps):
        return min(natural(requested_steps, 'requested_steps'), self.remaining_vector_steps)

    def record_after_step(self, actual_steps=1, *, actual_num_envs, warmup=False):
        natural(actual_steps, 'actual_steps', positive=True)
        natural(actual_num_envs, 'actual_num_envs', positive=True)
        if actual_num_envs != self.num_envs:
            raise ValueError('collector environment count changed')
        if actual_steps > self.remaining_vector_steps:
            raise ValueError('real simulation budget would be exceeded')
        self.vector_steps += actual_steps
        if warmup:
            self.warmup_vector_steps += actual_steps

    def state_dict(self):
        return {'schema': 1, 'num_envs': self.num_envs, 'macro_rounds': self.macro_rounds,
                'steps_per_round': self.steps_per_round, 'vector_steps': self.vector_steps,
                'warmup_vector_steps': self.warmup_vector_steps}

    @classmethod
    def restore(cls, state, *, expected_num_envs, expected_macro_rounds, expected_steps_per_round):
        if state.get('schema') != 1:
            raise ValueError('unsupported interaction accounting schema')
        if (state['num_envs'], state['macro_rounds'], state['steps_per_round']) != (
                expected_num_envs, expected_macro_rounds, expected_steps_per_round):
            raise ValueError('restored training budget does not match the configured budget')
        return cls(**{k: v for k, v in state.items() if k != 'schema'})

    def progress(self):
        return {'completed_macro_rounds': self.vector_steps//self.steps_per_round,
                'steps_into_next_macro_round': self.vector_steps % self.steps_per_round,
                'actual_vector_steps': self.vector_steps, 'warmup_included': self.warmup_vector_steps,
                'real_environment_transitions': self.real_transitions,
                'target_real_environment_transitions': self.num_envs*self.target_vector_steps,
                'budget_complete': self.remaining_vector_steps == 0}
