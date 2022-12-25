from abc import ABC, abstractmethod


class ObjectWithConf(ABC):
    """
    A class for objects with trackable configuration
    """

    @abstractmethod
    def get_conf(self):
        """
        Gets the instance configuration with the parameters the instance was initialised with
        :return: Dictionary
        """
        pass
