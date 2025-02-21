# Component 1: Input Validation
def _validate_numexpr_compatibility(op, op_str, left_op, right_op):
    return _can_use_numexpr(op, op_str, left_op, right_op, "evaluate")
# CCN: 1
# NLOC: 2
# Parameters: 4

# Component 2: Operand Preparation
def _prepare_operands(op, left_op, right_op):
    is_reversed = op.__name__.strip("_").startswith("r")
    if is_reversed:
        left_op, right_op = right_op, left_op
    return left_op, right_op, is_reversed
# CCN: 2
# NLOC: 4
# Parameters: 3

# Component 3: Core Evaluation
def _core_numexpr_eval(op_str, left_value, right_value):
    try:
        return ne.evaluate(
            f"left_value {op_str} right_value",
            local_dict={"left_value": left_value, "right_value": right_value},
            casting="safe",
        )
    except TypeError:
        return None
    except NotImplementedError as e:
        if _bool_arith_fallback(op_str, left_value, right_value):
            return None
        raise e
# CCN: 4 (try: +1, except TypeError: +1, except NotImplementedError: +1, if: +1)
# NLOC: 11
# Parameters: 3

# Main Function
def _evaluate_numexpr(op, op_str, left_op, right_op):
    if not _validate_numexpr_compatibility(op, op_str, left_op, right_op):
        return _evaluate_standard(op, op_str, left_op, right_op)

    left_value, right_value, is_reversed = _prepare_operands(op, left_op, right_op)
    result = _core_numexpr_eval(op_str, left_value, right_value)

    if is_reversed:
        left_op, right_op = right_op, left_op

    if _TEST_MODE:
        _store_test_result(result is not None)

    if result is None:
        result = _evaluate_standard(op, op_str, left_op, right_op)

    return result
# CCN: 5 (if not validate: +1, if is_reversed: +1, if _TEST_MODE: +1, if result is None: +1)
# NLOC: 12
# Parameters: 4