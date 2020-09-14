# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy as np

from .base_wake_combination import WakeCombination


class FractionalNorm(WakeCombination):
    """
    FractionalNorm applies a variable or fractional norm $n$ such that,

    $$
        u_tot = (u_wake^norm_order + u_field^norm_order)^(1/norm_order)
    $$
    """

    default_parameters = {"norm_order": 1.5, "wake_weight": 1.0}

    def __init__(self, parameter_dictionary):
        """
        Fractional Norm method to balance advantages of linear/sum-of-squares
        methods.

        Args:
            parameter_dictionary (dict): Model-specific parameters.
                Default values are used when a parameter is not included
                in `parameter_dictionary`. Possible key-value pairs include:

                -   **norm_order** (*dict*): The value used as the root and
                    power in the fractional norm.
        """
        super().__init__(parameter_dictionary)
        # self.logger = setup_logger(name=__name__)
        self.model_string = "fracnorm"
        model_dictionary = self._get_model_dict(__class__.default_parameters)
        self.norm_order = model_dictionary["norm_order"]
        self.wake_weight = model_dictionary["wake_weight"]

    def function(self, u_field, u_wake, turbnum=None):
        """
        Combines the base flow field with the velocity deficits
        using variable or fractional norm.

        Args:
            u_field (np.array): The base flow field.
            u_wake (np.array): The wake to apply to the base flow field.

        Returns:
            np.array: The resulting flow field after applying the wake to the
                base.
        """
        norm_order = self.norm_order[turbnum]
        return (u_wake**norm_order + u_field**norm_order)**(1 / norm_order)

    @property
    def norm_order(self):
        """
        The value used as the root and power in the fractional norm.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            norm_order (float): root/power in the fractional norm

        Returns:
            float: norm_order

        Raises:
            ValueError: Invalid value.
        """
        return self._norm_order

    @norm_order.setter
    def norm_order(self, value):
        # if type(value) is not np.array:  # or type(value) is not list:
        #     err_msg = ("Invalid value type given for " +
        #                "norm_order: {}, expected float.").format(value)
        #     self.logger.error(err_msg, stack_info=True)
        #     raise ValueError(err_msg)
        self._norm_order = value
        if value != __class__.default_parameters["norm_order"]:
            self.logger.info(
                ("Current value of norm_order, {0}, is not equal to tuned " +
                 "value of {1}.").format(
                     value, __class__.default_parameters["norm_order"]))

    @property
    def wake_weight(self):
        """
        The value used as the exponent to the number of wakes influencing a given turbine.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            wake_weight (float): exponent to nwakes

        Returns:
            float: wake_weight

        Raises:
            ValueError: Invalid value.
        """
        return self._wake_weight

    @wake_weight.setter
    def wake_weight(self, value):
        if type(value) is not float:
            err_msg = ("Invalid value type given for " +
                       "wake_weight: {}, expected float.").format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._wake_weight = value
        if value != __class__.default_parameters["wake_weight"]:
            self.logger.info(
                ("Current value of wake_weight, {0}, is not equal to tuned " +
                 "value of {1}.").format(
                     value, __class__.default_parameters["wake_weight"]))
