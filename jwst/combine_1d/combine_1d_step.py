#! /usr/bin/env python

from stdatamodels.jwst import datamodels

from ..stpipe import Step
from . import combine1d


__all__ = ["Combine1dStep"]


class Combine1dStep(Step):
    """

    Combine1dStep: Combine 1-D spectra

    """

    class_alias = "combine_1d"

    spec = """
    exptime_key = string(default="exposure_time") # use for weight
    sigma_clip = float(default=None) # Factor for clipping outliers
    """ # noqa: E501

    def process(self, input_file):

        with datamodels.open(input_file) as input_model:
            result = combine1d.combine_1d_spectra(input_model,
                                                  self.exptime_key,
                                                  sigma_clip=self.sigma_clip)

        return result
