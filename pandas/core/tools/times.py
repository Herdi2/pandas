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
    branches_reached[0] = 1    

    if errors not in ("raise", "coerce"):
        print("#01: Invalid errors argument.")
        branches_reached[1] = 1

        raise ValueError("errors must be one of 'raise', or 'coerce'.")

    def _convert_listlike(arg, format):
        print(f"#07: Entering _convert_listlike with arg={arg} and format={format}")
        branches_reached[7] = 1

        if isinstance(arg, (list, tuple)):
            print("#08: arg is a valid list or tuple instance.")
            branches_reached[8] = 1

            arg = np.array(arg, dtype="O")

        elif getattr(arg, "ndim", 1) > 1:
            print("#09: arg is of invalid type.")
            branches_reached[9] = 1

            raise TypeError(
                "arg must be a string, datetime, list, tuple, 1-d array, or Series"
            )

        arg = np.asarray(arg, dtype="O")

        if infer_time_format and format is None:
            print(f"#10: No format information and infer_time_format set to True, infers time with _guess_time_format_for_array with arg={arg}")
            branches_reached[10] = 1

            format = _guess_time_format_for_array(arg)

        times: list[time | None] = []
        if format is not None:
            print("#11: format is not None.")
            branches_reached[11] = 1

            for element in arg:
                try:
                    times.append(datetime.strptime(element, format).time())
                except (ValueError, TypeError) as err:
                    if errors == "raise":
                        print("#12: ValueError or TypeError occurred in try-except clause, and errors argument is set to \"raise\".")    
                        branches_reached[12] = 1

                        msg = (
                            f"Cannot convert {element} to a time with given "
                            f"format {format}"
                        )
                        raise ValueError(msg) from err
                    times.append(None)
        else:
            print(f"#13: No format information and infer format set to False. Prepare to find format in fixed _time_formats array.")
            branches_reached[13] = 1

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
                                branches_reached[16] = 1

                                # Put the found format in front
                                fmt = formats.pop(formats.index(time_format))
                                formats.insert(0, fmt)
                                format_found = True
                            break
                        except (ValueError, TypeError):
                            continue

                if time_object is not None:
                    times.append(time_object)
                elif errors == "raise":
                    print(f"#14: Unable to convert arg {arg} to a time, raises error according to errors argument.")
                    branches_reached[14] = 1

                    raise ValueError(f"Cannot convert arg {arg} to a time")
                else:
                    print(f"#15: Unable to convert arg {arg} to a time, appends None object rather than raising ValueError to errors argument.")
                    branches_reached[15] = 1

                    times.append(None)

        return times

    if arg is None:
        print("#02: arg is None, returns arg immediately.")
        branches_reached[2] = 1
        return arg
    elif isinstance(arg, time):
        print("#03: arg is already a time object, returns arg immediately.")
        branches_reached[3] = 1
        return arg
    elif isinstance(arg, ABCSeries):
        print("#04: arg is instance of ABCSeries, convert with _convert_listlike(arg._values,format)")
        branches_reached[4] = 1
        values = _convert_listlike(arg._values, format)
        return arg._constructor(values, index=arg.index, name=arg.name)
    elif isinstance(arg, ABCIndex):
        print("#05: arg is instance of ABCIndex, convert with _convert_listlike(arg,format).")
        branches_reached[5] = 1
        return _convert_listlike(arg, format)
    elif is_list_like(arg):
        print("#06: arg is list like according to is_list_like(arg), convert with _convert_listlike(arg, format).")
        branches_reached[6] = 1
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