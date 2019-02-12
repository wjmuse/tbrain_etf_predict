import pandas as pd
import matplotlib.pyplot as plt


def get_weekday(datestr, fmt='%Y%m%d'):
    """Get weekday from date str.
    Params:
        datestr: yyyymmdd(ex, "20150225"), type: str
    Return:
        weekday: from 1(Mon.) to 7(Sun.), type: int
    """
    return datetime.strptime(datestr, fmt).weekday() + 1


def evaluate_price(predict: pd.Series, actual: pd.Series, 
                   weight: list=[0.1, 0.15, 0.2, 0.25, 0.3], etf_code: str='') -> pd.Series:
    """Get score Series from evaluating price prediction.
    Params:
        predict: Each float value of predicted price, type: pd.Series with dtype: float
        actual: Each float value of target price, type: pd.Series with dtype: float
        weight: Weight value per day, type: float list
        etf_code: ETF code, type: str
    Return:
        result: Evaluated score per day, type: pd.Series with dtype: float
    """
    result = ((target - abs(predict - target)) / target) * 0.5
    result.name = etf_code
    return result


def evaluate_updown(predict: pd.Series, actual: pd.Series, etf_code: str='') -> pd.Series:
    """Get score Series from evaluating up-down prediction.
    Params:
        predict: Each value of predicted up-down 1, -1 or 0, type: pd.Series with dtype: int
        actual: Each value of target up-down 1, -1 or 0, type: pd.Series with dtype: int
        etf_code: ETF code, type: str
    Return:
        result: Evaluated score per day, type: pd.Series with dtype: float
    """
    result = 0.5 * (predict == actual)  # .apply(lambda x:  0.5 if x is True else 0)
    result.name = etf_code
    return result


def evaluate_score(predict_cprice: pd.Series, 
                   target_cprice: pd.Series,
                   predict_updown: pd.Series, 
                   target_updown: pd.Series,
                   weight: list=[0.1, 0.15, 0.2, 0.25, 0.3],
                   etf_code: str = "") -> float:
    """Get evaluate score.
    Params:
        ...
    Return:
        result: Sum evaluated score, type: float
    """
    price = evaluate_price(predict_cprice, target_cprice, wieht, etf_code)
    updown = evaluate_updown(predict_updown, target_updown, etf_code)
    return sum(price) + sum(updown)


def ez_plot(df: pd.DataFrame, figsize=(20, 10),
            title_prop=dict(label='Untitled', size=20),
            legend_prop=dict(loc='upper right', fontsize=24),
            **plot_kwargs):
    """Easy pyplot wrapper for dataframe.
    Params:
        df: Every columns in dataframe will be plotted.
        figsize: figure size.
        title_prop: Title property defined by pyplot api.
        legend_prop: Legend property defined by pyplot api.
        plot_kwargs: Other kwargs for pandas plot api, ex: `kind`: `line`, `bar`, `hist`, `box`, etc.
    Return:
        plt.figure
    """

    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    df.plot(ax=ax, **plot_kwargs)
    if ax.get_legend() and isinstance(legend_prop, dict):
        ax.legend(**legend_prop)
    if isinstance(title_prop, dict):
        ax.set_title(**title_prop)
    plt.close(fig)

    return fig

