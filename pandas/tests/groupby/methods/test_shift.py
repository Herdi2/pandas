
import pandas as pd
import pandas._testing as tm

from pandas.tests.config import coverage_wrapper

def test_shift_back_by_one(self):
    """Try shifting back by one"""
    df = pd.DataFrame({
        'group': ['Luleå', 'Luleå', 'Luleå', 'Stockholm', 'Stockholm'],
        'value': [4, 3, 9, 50, 20]
    })

    result = df.groupby('group')['value'].shift(-1)

    expected = pd.Series([3, 9, None, 20, None])

    assert tm.assert_series_equal(result, expected)

def test_shift_empty_arrays(self):
    df = pd.DataFrame({'group': [], 'value': []})

    result = df.groupby('group')['value'].shift(1)

    assert result.empty

def test_shift_into_empty(self):
    df = pd.DataFrame({
        'group': ['Luleå', 'Stockholm'],
        'value': [70, 120]
    })

    result = df.groupby('group')['value'].shift(1)

    expected = pd.Series([None, None])
    
    assert tm.assert_series_equal(result, expected)


def test_shift_invalid_periods_type(self):
    df = pd.DataFrame({'group': ['A'], 'value': [1]})
    caught = False
    try:
        with self.assertRaises(TypeError):
            df.groupby('group')['value'].shift("one")
    except:
        caught = True
    assert caught