from abc import ABC, abstractmethod


@dataclass
class GalaxyModel(ABC):
    r_list: List[float]
    Vtot_data: float
    Verr: float
    velocity_gas: float
    velocity_star: float
    velocity_bulge: float
    luminosity: float

    @abstractmethod
    def get_log_likelihood(theta, r_input, Vtot_data, Verr):
        pass

    @abstractmethod
    def get_log_prior(theta):
        pass

    @abstractmethod
    def get_log_probability(theta, r_input, Vtot_input, Verr):
        pass

