# -*- coding: utf-8 -*-

# utility
from metbit.utility.unipair import unipair
from metbit.utility.genpage import genpage

# preprocessing
from metbit.preprocessing.denoise_spec import decrease_noise
from metbit.preprocessing.normspec import Normalization

# multivariate
from metbit.multivariate.opls.anova import *
from metbit.multivariate.opls.base import *
from metbit.multivariate.opls.boxplot import *
from metbit.multivariate.opls.cross_validation import *
from metbit.multivariate.opls.elipse import *
from metbit.multivariate.opls.opls import *
from metbit.multivariate.opls.plotting import *
from metbit.multivariate.opls.pls import *
from metbit.multivariate.opls.pretreatment import *
from metbit.multivariate.opls.vip import *

# univariate
from metbit.univariate.lingress import lingress


# __all__ = ['unipair', 'genpage', 'decrease_noise', 'Normalization', 'anova', 'base', 'boxplot', 'cross_validation', 'elipse', 'opls', 'plotting', 'pls', 'pretreatment', 'vip', 'lingress']

