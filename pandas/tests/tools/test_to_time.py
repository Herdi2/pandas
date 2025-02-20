from datetime import time
import locale

import numpy as np
import pytest

from pandas.compat import PY311

from pandas import Series
import pandas._testing as tm
from pandas.core.tools.times import to_time

# The tests marked with this are locale-dependent.
# They pass, except when the machine locale is zh_CN or it_IT.
fails_on_non_english = pytest.mark.xfail(
    locale.getlocale()[0] in ("zh_CN", "it_IT"),
    reason="fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8",
    strict=False,
)


class TestToTime:
    @pytest.mark.parametrize(
        "time_string",
        [
            "14:15",
            "1415",
            pytest.param("2:15pm", marks=fails_on_non_english),
            pytest.param("0215pm", marks=fails_on_non_english),
            "14:15:00",
            "141500",
            pytest.param("2:15:00pm", marks=fails_on_non_english),
            pytest.param("021500pm", marks=fails_on_non_english),
            time(14, 15),
        ],
    )
    def test_parsers_time(self, time_string):
        # GH#11818
        assert to_time(time_string) == time(14, 15)

    def test_odd_format(self):
        new_string = "14.15"
        msg = r"Cannot convert arg \['14\.15'\] to a time"
        if not PY311:
            with pytest.raises(ValueError, match=msg):
                to_time(new_string)
        assert to_time(new_string, format="%H.%M") == time(14, 15)

    def test_arraylike(self):
        arg = ["14:15", "20:20"]
        expected_arr = [time(14, 15), time(20, 20)]
        assert to_time(arg) == expected_arr
        assert to_time(arg, format="%H:%M") == expected_arr
        assert to_time(arg, infer_time_format=True) == expected_arr
        assert to_time(arg, format="%I:%M%p", errors="coerce") == [None, None]

        with pytest.raises(ValueError, match="errors must be"):
            to_time(arg, format="%I:%M%p", errors="ignore")

        msg = "Cannot convert.+to a time with given format"
        with pytest.raises(ValueError, match=msg):
            to_time(arg, format="%I:%M%p", errors="raise")

        tm.assert_series_equal(
            to_time(Series(arg, name="test")), Series(expected_arr, name="test")
        )

        res = to_time(np.array(arg))
        assert isinstance(res, list)
        assert res == expected_arr

    # EXTENDED COVERAGE TEST CASES
    def test_None_arg(self):
        """Tests that arg=None, returns `None`. """
        arg = None
        assert to_time(arg) == arg

    def test_time_arg(self):
        """Tests that for arg of datetime.time object type, arg is returned."""
        arg = time(14,15)
        assert to_time(arg) == arg

    def test_matrix_arg(self):
        """Tests for expected TypeError when arg is a multidim numpy array."""
        arg = np.array([[1,3,4],[1,2,4]])
        msg = "arg must be a string, datetime, list, tuple, 1-d array, or Series"
        with pytest.raises(TypeError, match=msg):
            to_time(arg, format="%I:%M%p", errors="raise")

    def test_coerce_error(self):
        """Tests that with errors='coerce', ValueError is suppressed and None is returned for unparsable input string."""
        arg = "14.15"
        assert to_time(arg,format=None,errors='coerce') == None
