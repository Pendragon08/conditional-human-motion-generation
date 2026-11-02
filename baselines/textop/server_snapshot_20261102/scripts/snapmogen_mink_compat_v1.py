"""Adapt ScaleRetarget's six-position Mink call to Mink 0.0.13's keyword limits.

The original third-party file is not edited. This shim applies only in the
calling process and rejects ambiguous positional/keyword combinations.
"""
import inspect

class MinkLimitCompatibility:
    def __init__(self, module):
        self.module=module;self.original=module.solve_ik;self.corrected_calls=0
        names=list(inspect.signature(self.original).parameters)
        if names[:7]!=['configuration','tasks','dt','solver','damping','safety_break','limits']:
            raise ValueError('Unexpected Mink solve_ik signature')
        def adapted(*args,**kwargs):
            if len(args)==6 and isinstance(args[5],(list,tuple)):
                if 'limits' in kwargs or 'safety_break' in kwargs:raise TypeError('Ambiguous positional limits')
                self.corrected_calls+=1
                return self.original(*args[:5],limits=args[5],**kwargs)
            return self.original(*args,**kwargs)
        self.adapted=adapted
    def install(self):
        if self.module.solve_ik is not self.original:raise RuntimeError('Mink function changed before install')
        self.module.solve_ik=self.adapted
    def restore(self):
        if self.module.solve_ik is not self.adapted:raise RuntimeError('Mink function changed before restore')
        self.module.solve_ik=self.original
