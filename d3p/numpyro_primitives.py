import numpyro.primitives
from numpyro.primitives import Messenger, apply_stack, find_stack_level, CondIndepStackFrame, ExitStack, _PYRO_STACK
from contextlib import contextmanager
import numpyro
from numpyro import sample
import jax.numpy as jnp
from jax import lax
import jax
import warnings


import numpyro.handlers
class mask_observed(Messenger):
    """
    This messenger masks out some of the sample statements elementwise but only if the
    samples are observations. Additionally it applies scaling to ensure the expected sum over
    log-probabilities of the remaining samples is equal to that of the full batch.

    :param mask: a boolean or a boolean-valued array for masking elementwise log
        probability of sample sites (`True` includes a site, `False` excludes a site).
    """

    def __init__(self, fn=None, mask=True):
        if jnp.result_type(mask) != "bool":
            raise ValueError("`mask` should be a bool array.")
        self.mask = mask
        num_elems = jnp.sum(mask)
        batch_size = 1 if isinstance(mask, bool) else len(mask)
        self.scale = jnp.where(num_elems == 0, 0., batch_size / num_elems)
        super().__init__(fn)

    def process_message(self, msg):
        if msg.get("is_observed", False):
            if msg["type"] != "sample":
                if msg["type"] == "inspect":
                    msg["mask"] = (
                        self.mask if msg["mask"] is None else (self.mask & msg["mask"])
                    )
                return

            msg["fn"] = msg["fn"].mask(self.mask)
            msg["scale"] = self.scale if (prev_scale := msg.get("scale")) is None else prev_scale * self.scale


