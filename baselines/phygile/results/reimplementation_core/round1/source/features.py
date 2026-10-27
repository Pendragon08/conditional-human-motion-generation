"""Explicitly ordered 262D blocks; no implicit robot skeleton-index mapping."""
from dataclasses import dataclass
import numpy as np

BLOCKS = [('root_angular_velocity',3),('root_linear_velocity',3),('root_height',1),
          ('local_body_positions',36),('body_rotations_6d',174),('local_body_velocities',39),
          ('foot_contacts',4),('hand_contacts',2)]
NORMALIZED = np.r_[np.arange(43),np.arange(217,256)]

def pack_blocks(blocks):
    assert set(blocks)=={name for name,_ in BLOCKS}
    frames=None;arrays=[]
    for name,width in BLOCKS:
        value=np.asarray(blocks[name],dtype=np.float32)
        if value.ndim<2:raise ValueError(f'{name}: expected frame and feature axes')
        frames=value.shape[0] if frames is None else frames
        value=value.reshape(value.shape[0],-1)
        if value.shape!=(frames,width) or not np.isfinite(value).all():raise ValueError(name)
        if name.endswith('contacts') and not np.isin(value,[0,1]).all():raise ValueError(name+' must be binary')
        arrays.append(value)
    return np.concatenate(arrays,axis=-1)

@dataclass
class BlockNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, training_sequences):
        arrays=[np.asarray(x,dtype=np.float64) for x in training_sequences]
        if not arrays or any(x.ndim!=2 or x.shape[1]!=262 or not np.isfinite(x).all() for x in arrays):
            raise ValueError('Fit only on finite training [frames,262] sequences')
        joined=np.concatenate(arrays)
        if not len(joined):raise ValueError('No training frames')
        mean=np.zeros(262);std=np.ones(262)
        mean[NORMALIZED]=joined[:,NORMALIZED].mean(0)
        std[NORMALIZED]=np.maximum(joined[:,NORMALIZED].std(0),1e-6)
        return cls(mean.astype(np.float32),std.astype(np.float32))

    def transform(self,x):
        x=np.asarray(x,dtype=np.float32)
        if x.shape[-1]!=262:raise ValueError('Expected 262 features')
        return (x-self.mean)/self.std

    def inverse(self,x):return np.asarray(x,dtype=np.float32)*self.std+self.mean

def asfo_multipliers(sample_tags,rho_max):
    """Literal Eqs. 10–11; tags must be externally supplied, not invented here."""
    if rho_max<1 or not sample_tags or any(not tags for tags in sample_tags):raise ValueError('Nonempty tags required')
    tags=[set(x) for x in sample_tags]
    frequency={tag:sum(tag in x for x in tags) for tag in set().union(*tags)}
    target=float(np.median(list(frequency.values())))
    multipliers={tag:min(int(np.floor(target/freq+0.5)),rho_max) for tag,freq in frequency.items()}
    per_sample=np.array([max(multipliers[tag] for tag in x) for x in tags],dtype=np.int64)
    return per_sample,{'frequency':frequency,'median':target,'tag_multipliers':multipliers}
