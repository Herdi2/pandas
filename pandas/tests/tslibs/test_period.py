import numpy as np
import pytest
import pandas as pd
from datetime import datetime

from pandas._libs.tslibs import (
    iNaT,
    to_offset,
)
from pandas._libs.tslibs.period import (
    extract_ordinals,
    get_period_field_arr,
    period_asfreq,
    period_ordinal,
)

import pandas._testing as tm


def get_freq_code(freqstr: str) -> int:
    off = to_offset(freqstr, is_period=True)
    # error: "BaseOffset" has no attribute "_period_dtype_code"
    code = off._period_dtype_code  # type: ignore[attr-defined]
    return code


@pytest.mark.parametrize(
    "freq1,freq2,expected",
    [
        ("D", "h", 24),
        ("D", "min", 1440),
        ("D", "s", 86400),
        ("D", "ms", 86400000),
        ("D", "us", 86400000000),
        ("D", "ns", 86400000000000),
        ("h", "min", 60),
        ("h", "s", 3600),
        ("h", "ms", 3600000),
        ("h", "us", 3600000000),
        ("h", "ns", 3600000000000),
        ("min", "s", 60),
        ("min", "ms", 60000),
        ("min", "us", 60000000),
        ("min", "ns", 60000000000),
        ("s", "ms", 1000),
        ("s", "us", 1000000),
        ("s", "ns", 1000000000),
        ("ms", "us", 1000),
        ("ms", "ns", 1000000),
        ("us", "ns", 1000),
    ],
)
def test_intra_day_conversion_factors(freq1, freq2, expected):
    assert (
        period_asfreq(1, get_freq_code(freq1), get_freq_code(freq2), False) == expected
    )


@pytest.mark.parametrize(
    "freq,expected", [("Y", 0), ("M", 0), ("W", 1), ("D", 0), ("B", 0)]
)
def test_period_ordinal_start_values(freq, expected):
    # information for Jan. 1, 1970.
    assert period_ordinal(1970, 1, 1, 0, 0, 0, 0, 0, get_freq_code(freq)) == expected


@pytest.mark.parametrize(
    "dt,expected",
    [
        ((1970, 1, 4, 0, 0, 0, 0, 0), 1),
        ((1970, 1, 5, 0, 0, 0, 0, 0), 2),
        ((2013, 10, 6, 0, 0, 0, 0, 0), 2284),
        ((2013, 10, 7, 0, 0, 0, 0, 0), 2285),
    ],
)
def test_period_ordinal_week(dt, expected):
    args = dt + (get_freq_code("W"),)
    assert period_ordinal(*args) == expected


@pytest.mark.parametrize(
    "day,expected",
    [
        # Thursday (Oct. 3, 2013).
        (3, 11415),
        # Friday (Oct. 4, 2013).
        (4, 11416),
        # Saturday (Oct. 5, 2013).
        (5, 11417),
        # Sunday (Oct. 6, 2013).
        (6, 11417),
        # Monday (Oct. 7, 2013).
        (7, 11417),
        # Tuesday (Oct. 8, 2013).
        (8, 11418),
    ],
)
def test_period_ordinal_business_day(day, expected):
    # 5000 is PeriodDtypeCode for BusinessDay
    args = (2013, 10, day, 0, 0, 0, 0, 0, 5000)
    assert period_ordinal(*args) == expected


class TestExtractOrdinals:
    def test_extract_ordinals_raises(self):
        # with non-object, make sure we raise TypeError, not segfault
        arr = np.arange(5)
        freq = to_offset("D")
        with pytest.raises(TypeError, match="values must be object-dtype"):
            extract_ordinals(arr, freq)

    def test_extract_ordinals_2d(self):
        freq = to_offset("D")
        arr = np.empty(10, dtype=object)
        arr[:] = iNaT

        res = extract_ordinals(arr, freq)
        res2 = extract_ordinals(arr.reshape(5, 2), freq)
        tm.assert_numpy_array_equal(res, res2.reshape(-1))


def test_get_period_field_array_raises_on_out_of_range():
    msg = "Buffer dtype mismatch, expected 'const int64_t' but got 'double'"
    with pytest.raises(ValueError, match=msg):
        get_period_field_arr(-1, np.empty(1), 0)

# 10 test cases for ordinal dates
@pytest.mark.parametrize(
    "ordinal_str,expected_date_str",
    [
        # Basic ordinal date tests
        ("2022-001", "2022-01-01"),  # First day of year
        ("2022-032", "2022-02-01"),  # February 1
        ("2022-219", "2022-08-07"),  # August 7 (from row 6)
        ("2022-365", "2022-12-31"),  # Last day of common year
        
        # Leap year tests
        ("2020-060", "2020-02-29"),  # Feb 29 in leap year
        ("2020-366", "2020-12-31"),  # Last day of leap year
        
        # 24th century ordinal dates
        ("2300-180", "2300-06-29"),  # Mid-year in 24th century
        ("2320-001", "2320-01-01"),  # First day of 2320
        ("2400-060", "2400-02-29"),  # Feb 29 in leap year (div by 400)
        ("2400-366", "2400-12-31"),  # Last day of leap year 2400
    ],
)
def test_period_with_ordinal_dates(ordinal_str, expected_date_str):
    """Test that Period constructor can parse ISO 8601 ordinal dates."""
    # Create Period with ordinal date string
    period = pd.Period(ordinal_str)
    
    # Convert expected date string to datetime for comparison
    expected_date = datetime.strptime(expected_date_str, "%Y-%m-%d")
    
    # Verify date is correctly parsed
    assert period.year == expected_date.year, f"Year mismatch for {ordinal_str}"
    assert period.month == expected_date.month, f"Month mismatch for {ordinal_str}"
    assert period.day == expected_date.day, f"Day mismatch for {ordinal_str}"


# 10 test cases for 24th century weeks
@pytest.mark.parametrize(
    "week_str,expected_end_date",
    [
        # 24th century weekly tests 
        ("2301-01-01/2301-01-07", "2301-01-07"),  # First week of 24th century
        ("2350-06-25/2350-07-01", "2350-07-01"),  # Mid-24th century
        ("2399-12-25/2399-12-31", "2399-12-31"),  # Last week of 24th century
        
        # Week crossing years in future centuries - modified to use end dates
        ("2361-12-31/2362-01-06", "2362-01-06"),  # Week crossing years
        ("2481-12-29/2482-01-04", "2482-01-04"),  # Week from row 73
        
        # Problem cases from the issue table - modified to use end dates
        ("2061-12-26/2062-01-01", "2062-01-01"),  # Row 59
        ("2181-12-31/2182-01-06", "2182-01-06"),  # Row 63
        ("2272-01-01/2272-01-07", "2272-01-07"),  # Row 66
        ("2362-01-01/2362-01-07", "2362-01-07"),  # Row 69
        ("2452-01-01/2452-01-07", "2452-01-07"),  # Row 72
    ],
)
def test_period_with_24th_century_weeks(week_str, expected_end_date):
    """Test that Period constructor can handle week format strings in the 24th century."""
    # Create Period with week string
    period = pd.Period(week_str)
    
    # Get the end date from the string
    _, end_str = week_str.split('/')
    end_date = pd.Timestamp(end_str)
    
    # Verify the period's day matches the end date's day
    assert period.day == end_date.day, f"Day mismatch for {week_str}"
    assert period.month == end_date.month, f"Month mismatch for {week_str}"
    assert period.year == end_date.year, f"Year mismatch for {week_str}"

@pytest.mark.parametrize(
    "problematic_year",
    [
        2060,  # 60s decade
        2070,  # 70s decade
        2080,  # 80s decade
        2090,  # 90s decade
        2172,  # > 21:59
        2272,  # Specifically mentioned in issue (interpreted as 22:72)
        2362,  # > 23:59
        2400,  # 24:00 edge case
        2482,  # From the specific example (24:82)
    ]
)
def test_problematic_years_roundtrip(problematic_year):
    """
    Test string conversion for years that could be misinterpreted as hour:minute.
    """
    # 1. Create Period directly with the year
    original = pd.Period(freq='W', year=problematic_year)
    
    # 2. Convert to string
    period_str = str(original)
    print(f"\nPeriod(freq='W', year={problematic_year}) -> '{period_str}'")
    
    # 3. Recreate from string 
    # Previously, years like 2272 would be interpreted as 22:72 (hours:minutes) 
    recreated = pd.Period(period_str)
    
    # 4. Verify the year is correct
    assert recreated.year == problematic_year, f"Year mismatch: {recreated.year} != {problematic_year}"


@pytest.mark.parametrize(
    "datestring,expected", # expected = [str,freqstr,start_time,end_time,day_of_week,week]
    [   # basic input
        ('20250106-20250112',['2025-01-06/2025-01-12', 'W-SUN', '2025-01-06 00:00:00', '2025-01-12 23:59:59.999999999', 6, 2]), # mon-sun 
        # week turning over
        ('20250101-20250107',['2025-01-01/2025-01-07', 'W-TUE', '2025-01-01 00:00:00', '2025-01-07 23:59:59.999999999', 1, 2]), # wed-tue
        ('20250112-20250118',['2025-01-12/2025-01-18', 'W-SAT', '2025-01-12 00:00:00', '2025-01-18 23:59:59.999999999', 5, 3]), # sun-sat

        # month turning over
        ('20250228-20250306',['2025-02-28/2025-03-06', 'W-THU', '2025-02-28 00:00:00', '2025-03-06 23:59:59.999999999', 3, 10]), # fri-thur
        ('20251125-20251201',['2025-11-25/2025-12-01', 'W-MON', '2025-11-25 00:00:00','2025-12-01 23:59:59.999999999', 0, 49]), # tue-mon

        # year turning over
        ('19991231-20000106', ['1999-12-31/2000-01-06', 'W-THU', '1999-12-31 00:00:00', '2000-01-06 23:59:59.999999999', 3, 1]), # fri-thur
        ('20161229-20170104', ['2016-12-29/2017-01-04', 'W-WED', '2016-12-29 00:00:00', '2017-01-04 23:59:59.999999999', 2, 1]), # thur-wed
        ('20121231-20130106', ['2012-12-31/2013-01-06', 'W-SUN', '2012-12-31 00:00:00', '2013-01-06 23:59:59.999999999', 6, 1]), # mon-sun 
        # leap day
        ('20240226-20240303', ['2024-02-26/2024-03-03', 'W-SUN', '2024-02-26 00:00:00', '2024-03-03 23:59:59.999999999', 6, 9]), # mon-sun
    ],
)
def test_period_parse_weeks_positive(datestring,expected):
    """
    Tests correct attributes for Period objects created from the
    dedicated YYYYMMDD-YYYYMMDD week format with valid inputs.
    """
    p = pd.Period(datestring)
    assert str(p) == expected[0]
    assert p.freqstr == expected[1]
    assert str(p.start_time) == expected[2]
    assert str(p.end_time) == expected[3]
    assert p.day_of_week == expected[4]
    assert p.week == expected[5]

@pytest.mark.parametrize(
    "datestring",
    [   # basic input
        '20250106-20250112', # mon-sun 
        # week turning over
        '20250101-20250107', # wed-tue
        '20250112-20250118', # sun-sat

        # month turning over
        '20250228-20250306', # fri-thur
        '20251125-20251201', # tue-mon

        # year turning over
        '19991231-20000106',
        '20161229-20170104',
        '20121231-20130106',
    ],
)
def test_period_parse_weeks_equivalent(datestring):
    """
    Checks that string representation of objects created with the 
    YYYYMMDD-YYYYMMDD format can successfully be fed back into Period
    to create the equivalent week period.
    """
    p = pd.Period(datestring)
    p_recreation = pd.Period(str(p))
    assert p == p_recreation

@pytest.mark.parametrize(
    "datestring,expected",
    [
        # Same start and end date
        ('20131231-20131231', ValueError),

        # More or less than 7 days in span
        ('20130101-20130104', ValueError),
        ('20121231-20130102', ValueError),
        ('20121231-20130131', ValueError),
        # Inverse order

        ('20120107-20120101', ValueError),
        ('20120101-20111226', ValueError),

        # Out of bounds dates
        ('20250229-20250306', Exception),
        ('20251229-20251306', Exception),

        # Faulty formats
        ('251229-260106', Exception), # Two-digit years representation
        ('990303-990309', Exception), # Two-digit year representation
        ('2000912-20010918', Exception), # Missing digits
        ('20250106- 20250112', Exception) # Extra whitespace
    ],
)
def test_period_parse_weeks_err(datestring,expected):
    """
    Checks that the expected errors are thrown when attempting to
    create a Period objects from the string format YYYYMMDD-YYYYMMDD with
    invalid inputs.
    """
    with pytest.raises(expected):
        pd.Period(datestring)

@pytest.mark.parametrize(
    "datestring,expected",
    [
        ("1970-1973", [[1970, 1971, 1972, 1973], 'Y-DEC']),
        ("1867-1869", [[1867, 1868, 1869], 'Y-DEC']),
        ("1849-1851", [[1849, 1850, 1851], 'Y-DEC']),
        ("1959-1963", [[1959, 1960, 1961, 1962, 1963], 'Y-DEC'])
        
    ]
)
def test_period_parse_years_retro(datestring,expected):
    """
    Test intialisation of the Period object using the YYYY-YYYY format
    The test is validated by looking at the frequency string parameter and that each value
    in the timespan is initialised correctly
    """
    p = pd.Period(datestring)
    assert (p.year == pd.Index(expected[0])).all()# Test that year values match
    assert p.freqstr == expected[1]# Check the frequency
    
@pytest.mark.parametrize(
    "datestring,expected",
    [
        # Test year switch in 1960s
        ("1960Q3-1961Q1", [[1960, 1960, 1961], [3, 4, 1], 'Q-DEC']),
        # Test year switch in 1830s
        ("1830Q1-1831Q2", [[1830, 1830, 1830, 1830, 1831, 1831], [1, 2, 3, 4, 1, 2], 'Q-DEC']),
        # Test year switch in 1980s
        ("1980Q4-1981Q3", [[1980, 1981, 1981, 1981], [4, 1, 2, 3], 'Q-DEC']),
        # Test year switch in 1850s
        ("1850Q2-1851Q1", [[1850, 1850, 1850, 1851], [2, 3, 4, 1], 'Q-DEC']),
        # Test dacade switch in 1800s
        ("1859Q2-1860Q1", [[1859, 1859, 1859, 1860], [2, 3, 4, 1], 'Q-DEC']),
        # Test dacade switch in 1900s
        ("1929Q3-1930Q2", [[1929, 1929, 1930, 1930], [3, 4, 1, 2], 'Q-DEC'])
    ],
)
def test_period_parse_quarters_retro(datestring,expected):
    """
    Test Quarter format YYYYQA-YYYYQB for Period() object initialisation.
    We validate the test by looking at the year and quarter of each value ntry in the timespan
    """
    p = pd.Period(datestring)
    assert (p.year == pd.Index(expected[0])).all()# Test that year values match
    assert (p.quarter == pd.Index(expected[1])).all()# Test that quarter values match
    assert p.freqstr == expected[2]# Check that the frequency
    
@pytest.mark.parametrize(
    "datestring, expected",
    [
        # Period ends on same day as starts in 1960s
        ("19600703-19600703", ValueError),
        # Incorrect formatted years parameters and in 1970s
        ("1970-05-04-1970-05-02", ValueError)
    ],
)
def test_old_and_retro_periods_errors(datestring, expected):
    """
    We try initialising an invalid Period() object by using invalid parameters.
    We expect our method to return a ValueError upon encountering such parameters.
    """
    with pytest.raises(expected):
        pd.Period(datestring)
        
@pytest.mark.parametrize(
    "datestring, expected",
    [
        #Finland's independence
        ("19171206-19171212", 2),# From Thursday to Wednesday
        #Year turning over in 1960s
        ("19681230-19690105", 6),#From Monday to Sunday
        #Leap day in 1800s
        ("18640225-18640302", 2),# From Thursday to Wednesday
        #Leap day in 1900s
        ("19800226-19800303", 0),# From Tuesday to Monday
        #Decade switch in 1800s
        ("18191230-18200105",2),# Thursday to Wednesday
        #Decade switch from 1940s to 1950s
        ("19491228-19500103", 1)# Wednesday Tuesday
    ],
)
def test_old_and_retro_periods(datestring, expected):
    """
    Test rcreating the period object using the YYYYMMDD-YYYYMMDD format.
    We verify our test by looking at which day of the week we end on.
    We test in two different centuries as well as edge cases where we have
    a year shift or a decade shift.
    """
    p = pd.Period(datestring)
    assert p.day_of_week == expected
    
@pytest.mark.parametrize(
    "datestring",
    [
        # The black friday stock market week.
        "18690924-18690930",
        #Finland's independence
        "19171206-19171212",
        #Year turning over in 1960s
        "19681230-19690105",
        #Year turning over in 1970s
        "19751229-19760104",
        #Year turning over in 1990s
        "19931231-19940106",
        #Decade switch from 1940s to 1950s
        "19491228-19500103",
        #Decade switch in 1800s
        "18191230-18200105",
        #Year switch in 1800s
        "18531229-18540104",
        #Leap day in 1800s
        "18640225-18640302",
        #Leap day in 1900s
        "19800226-19800303",
    ],
)
def test_old_and_retro_periods_line_formatted_reinitialised(datestring):
    """
    Test re-creating the period object using the YYYYMMDD-YYYYMMDD format and making sure
    we get the same object in return. We test in various edge cases such as 1900s and 1800s, decade switches,
    as well as year swithes
    """
    p = pd.Period(datestring)
    p_recreation = pd.Period(str(p))
    assert p == p_recreation