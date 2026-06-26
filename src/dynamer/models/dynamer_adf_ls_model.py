"""DynaMER-ADF-LS model wrapper.

This variant reuses the DynaMER-ADF implementation with the low-spike / LS configuration
controlled through the corresponding training configuration.
"""

from src.dynamer.models.dynamer_adf_model import DynaMERADFModel

DynaMERADFLSModel = DynaMERADFModel

# Backward-compatible aliases
DynaMERv4Model = DynaMERADFLSModel
DynaMERv3Model = DynaMERADFLSModel
