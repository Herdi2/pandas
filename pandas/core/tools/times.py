from __future__ import annotations

from datetime import (
    datetime,
    time,
)
from typing import TYPE_CHECKING

import numpy as np

from pandas._libs.lib import is_list_like

from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import notna

if TYPE_CHECKING:
    from pandas._typing import DateTimeErrorChoices

import pytest

from pandas.compat import PY311
from pandas import Series
import pandas._testing as tm

def to_time(
    arg,
    format: str | None = None,
    infer_time_format: bool = False,
    errors: DateTimeErrorChoices = "raise",
):
    """
    Parse time strings to time objects using fixed strptime formats ("%H:%M",
    "%H%M", "%I:%M%p", "%I%M%p", "%H:%M:%S", "%H%M%S", "%I:%M:%S%p",
    "%I%M%S%p")

    Use infer_time_format if all the strings are in the same format to speed
    up conversion.

    Parameters
    ----------
    arg : string in time format, datetime.time, list, tuple, 1-d array,  Series
    format : str, default None
        Format used to convert arg into a time object.  If None, fixed formats
        are used.
    infer_time_format: bool, default False
        Infer the time format based on the first non-NaN element.  If all
        strings are in the same format, this will speed up conversion.
    errors : {'raise', 'coerce'}, default 'raise'
        - If 'raise', then invalid parsing will raise an exception
        - If 'coerce', then invalid parsing will be set as None

    Returns
    -------
    datetime.time
    """
    print(f"#00: Entering to_time() with arg={arg}, format={format}, infer_time_format={infer_time_format}, errors={errors}")

    if errors not in ("raise", "coerce"):
        print("#01: Invalid errors argument.")
        record_branch(1)

        raise ValueError("errors must be one of 'raise', or 'coerce'.")

    def _convert_listlike(arg, format):
        print(f"#07: Entering _convert_listlike with arg={arg} and format={format}")
        record_branch(7)

        if isinstance(arg, (list, tuple)):
            print("#08: arg is a valid list or tuple instance.")
            record_branch(8)

            arg = np.array(arg, dtype="O")

        elif getattr(arg, "ndim", 1) > 1:
            print("#09: arg is of invalid type.")
            record_branch(9)

            raise TypeError(
                "arg must be a string, datetime, list, tuple, 1-d array, or Series"
            )

        arg = np.asarray(arg, dtype="O")

        if infer_time_format and format is None:
            print(f"#10: No format information and infer_time_format set to True, infers time with _guess_time_format_for_array with arg={arg}")
            record_branch(10)

            format = _guess_time_format_for_array(arg)

        times: list[time | None] = []
        if format is not None:
            print("#11: format is not None.")
            record_branch(11)

            for element in arg:
                try:
                    times.append(datetime.strptime(element, format).time())
                except (ValueError, TypeError) as err:
                    if errors == "raise":
                        print("#12: ValueError or TypeError occurred in try-except clause, and errors argument is set to \"raise\".")    
                        record_branch(12)

                        msg = (
                            f"Cannot convert {element} to a time with given "
                            f"format {format}"
                        )
                        raise ValueError(msg) from err
                    times.append(None)
        else:
            print(f"#13: No format information and infer format set to False. Prepare to find format in fixed _time_formats array.")
            record_branch(13)

            formats = _time_formats[:]
            format_found = False
            for element in arg:
                time_object = None
                try:
                    time_object = time.fromisoformat(element)
                except (ValueError, TypeError):
                    for time_format in formats:
                        try:
                            time_object = datetime.strptime(element, time_format).time()
                            if not format_found:
                                print(f"#16: format_found previously False, set to True, since format was found in _time_formats.")
                                record_branch(16)

                                # Put the found format in front
                                fmt = formats.pop(formats.index(time_format))
                                formats.insert(0, fmt)
                                format_found = True
                            break
                        except (ValueError, TypeError):
                            continue

                if time_object is not None:
                    print(f"17: time_object not None, appending to times array.")
                    record_branch(17)
                    times.append(time_object)
                elif errors == "raise":
                    print(f"#14: Unable to convert arg {arg} to a time, raises error according to errors argument.")
                    record_branch(14)

                    raise ValueError(f"Cannot convert arg {arg} to a time")
                else:
                    print(f"#15: Unable to convert arg {arg} to a time, appends None object rather than raising ValueError to errors argument.")
                    record_branch(15)

                    times.append(None)

        return times

    if arg is None:
        print("#02: arg is None, returns arg immediately.")
        record_branch(2)
        return arg
    elif isinstance(arg, time):
        print("#03: arg is already a time object, returns arg immediately.")
        record_branch(3)
        return arg
    elif isinstance(arg, ABCSeries):
        print("#04: arg is instance of ABCSeries, convert with _convert_listlike(arg._values,format)")
        record_branch(4)
        values = _convert_listlike(arg._values, format)
        return arg._constructor(values, index=arg.index, name=arg.name)
    elif isinstance(arg, ABCIndex):
        print("#05: arg is instance of ABCIndex, convert with _convert_listlike(arg,format).")
        record_branch(5)
        return _convert_listlike(arg, format)
    elif is_list_like(arg):
        print("#06: arg is list like according to is_list_like(arg), convert with _convert_listlike(arg, format).")
        record_branch(6)
        return _convert_listlike(arg, format)

    return _convert_listlike(np.array([arg]), format)[0]


# Fixed time formats for time parsing
_time_formats = [
    "%H:%M",
    "%H%M",
    "%I:%M%p",
    "%I%M%p",
    "%H:%M:%S",
    "%H%M%S",
    "%I:%M:%S%p",
    "%I%M%S%p",
]


def _guess_time_format_for_array(arr):
    # Try to guess the format based on the first non-NaN element
    non_nan_elements = notna(arr).nonzero()[0]
    if len(non_nan_elements):
        element = arr[non_nan_elements[0]]
        for time_format in _time_formats:
            try:
                datetime.strptime(element, time_format)
                return time_format
            except ValueError:
                pass

    return None

# Global array for tracking branch access.
branches_reached = [0]*17
print("Array branches_reached initiated.")

def run_tests(extended=False):
    """Manual import and execution of pytest test cases for coverage testing."""
    # EXISTING TEST CASES
    def test_parsers_time(time_string):
        assert to_time(time_string) == time(14, 15)

    def test_odd_format():
        new_string = "14.15"
        msg = r"Cannot convert arg \['14\.15'\] to a time"
        if not PY311:
            with pytest.raises(ValueError, match=msg):
                to_time(new_string)
        assert to_time(new_string, format="%H.%M") == time(14, 15)

    def test_arraylike():
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
    def test_None_arg():
        arg = None
        assert to_time(arg) == arg

    def test_time_arg():
        arg = time(14,15)
        assert to_time(arg) == arg

    def test_matrix_arg():
        arg = np.array([[1,3,4],[1,2,4]])
        msg = "arg must be a string, datetime, list, tuple, 1-d array, or Series"
        with pytest.raises(TypeError, match=msg):
            to_time(arg, format="%I:%M%p", errors="raise")
  
    def test_coerce_error():
        arg = "14.15"
        assert to_time(arg,format=None,errors='coerce') == None

    # RUN EXISTING TEST CASES
    time_strings = ["14:15",
                "1415",
                "2:15pm",
                "0215pm",
                "14:15:00",
                "141500",
                "2:15:00pm",
                "021500pm",
                time(14, 15)]
    for entry in time_strings:
        test_parsers_time(entry)
    test_odd_format()
    test_arraylike()

    if extended == True:    
        # RUN EXTENDED COVERAGE
        test_None_arg()
        test_time_arg()
        test_matrix_arg()
        test_coerce_error()

def record_branch(branch_id):
    """Track execution count of a specific branch."""
    if branch_id in branch_coverage:
        branch_coverage[branch_id] += 1
    else:
        branch_coverage[branch_id] = 1

def finalize_coverage():
    """After all test cases execute, compute and save the coverage report."""
    save_coverage()  # Save coverage once all tests finish
    print("\n Final Manual Coverage Report Generated!")

def save_coverage():
    """Compute coverage percentage and save report."""
    total_branches = max(branch_coverage.keys(), default=0)  # Prevent KeyError if empty
    executed_branches = len([b for b, count in branch_coverage.items() if count > 0])
    coverage_score = (executed_branches / total_branches) * 100 if total_branches > 0 else 0
    for branch, count in sorted(branch_coverage.items()):
        print(f"Branch {branch}: executed {count} times\n")
    print(f"Manual Coverage Score: {coverage_score:.2f}%")  # Print to console

# Global dictionary to track which branches are executed
branch_coverage = {}

if __name__ == '__main__':
    # Test coverage and print results
    run_tests(extended=False)
    finalize_coverage()