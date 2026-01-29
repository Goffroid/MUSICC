# src/__init__.py
"""
Пакет для генерации музыки с помощью ансамблевых методов
"""

from .data_preprocessing import MIDIDataPreprocessor
from .feature_engineering import FeatureEngineer
from .models import BaseModels, AdvancedModels, ModelEvaluator
from .ensemble_methods import EnsembleMethods

__version__ = "1.0.0"
__author__ = "Music Generation Project"
__all__ = [
    'MIDIDataPreprocessor',
    'FeatureEngineer', 
    'BaseModels',
    'AdvancedModels',
    'ModelEvaluator',
    'EnsembleMethods'
]