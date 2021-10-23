from functools import partial

from src.base_model import BaseModel
from src.utils import pseudoinverse, gradient


class DampedGN(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calculate_update_direction(self, param, x, y):
        errors = self._calculate_errors(param, x, y)
        f = partial(self._calculate_errors, x=x, y=y)
        jacobian = gradient(param, f)
        return pseudoinverse(jacobian) @ errors
