# -*- coding: utf-8 -*-

# utility
from utility.unipair import unipair
from utility.genpage import gen_page

# preprocessing
from preprocessing.denoise_spec import decrease_noise
from preprocessing.normspec import Normalization

# multivariate
from multivariate.opls.anova import *
from multivariate.opls.base import *
from multivariate.opls.boxplot import *
from multivariate.opls.cross_validation import *
from multivariate.opls.elipse import *
from multivariate.opls.opls import *
from multivariate.opls.plotting import *
from multivariate.opls.pls import *
from multivariate.opls.pretreatment import *
from multivariate.opls.vip import *

# univariate
from univariate.lingress import lingress


# __all__ = ['unipair', 'genpage', 'decrease_noise', 'Normalization', 'anova', 'base', 'boxplot', 'cross_validation', 'elipse', 'opls', 'plotting', 'pls', 'pretreatment', 'vip', 'lingress']

