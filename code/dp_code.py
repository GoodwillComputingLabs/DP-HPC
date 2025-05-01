import pydp as dp
from pydp.algorithms.laplacian import BoundedSum, BoundedMean, BoundedStandardDeviation, BoundedVariance, Max, Min, Count, Median
import diffprivlib.tools as ibm

########################################################################
# PyDP queries
########################################################################
def pydp_mean(df, privacy_budget: float, low=1, up=10, t="float"):
    x = BoundedMean(epsilon=privacy_budget, lower_bound=low, upper_bound=up, dtype=t)
    return x.quick_result(df.to_list())

def pydp_sum(df, privacy_budget: float, low=1, up=10, t="float"):
    x = BoundedSum(epsilon=privacy_budget, l0_sensitivity = 1, lower_bound=low, upper_bound=up, dtype=t)
    return x.quick_result(df.to_list())

def pydp_variance(df, privacy_budget: float, low=1, up=10, t="float"):
    x = BoundedVariance(epsilon=privacy_budget, lower_bound=low, upper_bound=up, dtype=t)
    return x.quick_result(df.to_list())

def pydp_min(df, privacy_budget: float, low=1, up=10, t="float"):
    x = Min(epsilon=privacy_budget, lower_bound=low, upper_bound=up, dtype=t)
    return x.quick_result(df.to_list())

def pydp_max(df, privacy_budget: float, low=1, up=10, t="float"):
    x = Max(epsilon=privacy_budget, lower_bound=low, upper_bound=up, dtype=t, l0_sensitivity=2, linf_sensitivity=1)
    return x.quick_result(df.to_list())

def pydp_count(df, privacy_budget: float, low=1, up=10, t="float"):
    x = Count(epsilon=privacy_budget, dtype=t)
    return x.quick_result(df.to_list())

def pydp_median(df, privacy_budget: float, low=1, up=10, t="float"):
    x = Median(epsilon=privacy_budget, lower_bound=low, upper_bound=up, dtype=t)
    return x.quick_result(df.to_list())

def pydp_std(df, privacy_budget: float, low=1, up=10, t="float"):
    x = BoundedStandardDeviation(epsilon=privacy_budget, lower_bound=low, upper_bound=up, dtype=t)
    return x.quick_result(df.to_list())

########################################################################
# diffpriv queries
########################################################################
def diffpriv_mean(df, privacy_budget: float, low=1, up=10, t=float):
    x = ibm.mean(df, epsilon=privacy_budget, bounds=(low, up), dtype=t)
    return x

def diffpriv_sum(df, privacy_budget: float, low=1, up=10, t=int):
    x = ibm.sum(df, epsilon=privacy_budget, bounds=(low, up), dtype=t)
    return x

def diffpriv_std(df, privacy_budget: float, low=1, up=10, t=float):
    x = ibm.std(df, epsilon=privacy_budget, bounds=(low, up), dtype=t)
    return x

def diffpriv_var(df, privacy_budget: float, low=1, up=10, t=float):
    x = ibm.var(df, epsilon=privacy_budget, bounds=(low, up), dtype=t)
    return x

def diffpriv_median(df, privacy_budget: float, low=1, up=10, t=float):
    x = ibm.median(df, epsilon=privacy_budget, bounds=(low, up), dtype=t)
    return x

########################################################################
# Function to compute the DP queries
########################################################################
def dp_global(library, ite, e, df, op="MEAN", low=1, up=10):

    result = 0
    if(library == "PYDP"):
        if(op == "MEAN"):
            result = pydp_mean(df, e, low, up)
        elif(op == "SUM"):
            result = pydp_sum(df, e, low, up)
        elif(op == "STD"):
            result = pydp_std(df, e, low, up)
        elif(op == "VAR"):
            result = pydp_variance(df, e, low, up)
        elif(op == "MIN"):
            result = pydp_min(df, e, low, up)
        elif(op == "MAX"):
            result = pydp_max(df, e, low, up)
        elif(op == "COUNT"):
            result = pydp_count(df, e, low, up)
        elif(op == "MEDIAN"):
            result = pydp_median(df, e, low, up)
    else:
        if(op == "MEAN"):
            result = diffpriv_mean(df, e, low, up)
        elif(op == "SUM"):
            result = diffpriv_sum(df, e, low, up)
        elif(op == "STD"):
            result = diffpriv_std(df, e, low, up)
        elif(op == "VAR"):
            result = diffpriv_var(df, e, low, up)
        elif(op == "MEDIAN"):
            result = diffpriv_median(df, e, low, up)

    return result