"""
This module contains generic trainer functionalities, added as a Mixin to 
the Trainer module.
"""

import inspect


class BaseTrainerMixin:
    """
    Base trainer mix in class containing universal trainer functionalities.

    Parts of the implementation are adapted from
    https://github.com/scverse/scvi-tools/blob/master/scvi/model/base/_base_model.py#L63
    (01.10.2022).
    """
    def _get_user_attributes(self) -> list:
        """
        Get all the attributes defined in a trainer instance.

        Returns
        ----------
        attributes:
            Attributes defined in a trainer instance.
        """
        attributes = inspect.getmembers(
            self, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if not (
            a[0].startswith("__") and a[0].endswith("__"))]
        return attributes

    def _get_public_attributes(self) -> dict:
        """
        Get only public attributes defined in a trainer instance. By convention
        public attributes have a trailing underscore.

        Returns
        ----------
        public_attributes:
            Public attributes defined in a trainer instance.
        """
        public_attributes = self._get_user_attributes()
        public_attributes = {a[0]: a[1] for a in public_attributes if 
                             a[0][-1] == "_"}
        return public_attributes