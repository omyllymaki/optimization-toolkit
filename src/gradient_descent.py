from src.base_model import BaseModel
from src.utils import gradient


class GD(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calculate_update_direction(self, param, x, y):
        cost_calculation = lambda p: self._calculate_cost(self._calculate_errors(p, x, y))
        return gradient(param, cost_calculation)
