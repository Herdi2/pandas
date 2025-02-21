# Component 1: Input Validation
def _validate_numexpr_compatibility(op, op_str, left_op, right_op):
    return _can_use_numexpr(op, op_str, left_op, right_op, "evaluate")
# CCN: 1, NLOC: 2, Parameters: 4

# Component 2: Operand Preparation
def _prepare_operands(op, left_op, right_op):
    is_reversed = op.__name__.strip("_").startswith("r")
    if is_reversed:
        left_op, right_op = right_op, left_op
    return left_op, right_op, is_reversed
# CCN: 2, NLOC: 4, Parameters: 3

# Component 3: Core Evaluation
def _core_numexpr_eval(op_str, left_value, right_value):
    return ne.evaluate(
        f"left_value {op_str} right_value",
        local_dict={"left_value": left_value, "right_value": right_value},
        casting="safe",
    )
# CCN: 1, NLOC: 6, Parameters: 3

# Component 4: Error Management
def _handle_evaluation_errors(result, op_str, left_op, right_op):
    if result is None:
        return False
    try:
        return True, result
    except TypeError:
        return False, None
    except NotImplementedError:
        if _bool_arith_fallback(op_str, left_op, right_op):
            return False, None
        raise
# CCN: 5, NLOC: 10, Parameters: 4

# Main Function
def _evaluate_numexpr(op, op_str, left_op, right_op):
    result = None

    if not _validate_numexpr_compatibility(op, op_str, left_op, right_op):
        return _evaluate_standard(op, op_str, left_op, right_op)

    left_value, right_value, is_reversed = _prepare_operands(op, left_op, right_op)

    try:
        result = _core_numexpr_eval(op_str, left_value, right_value)
        success, eval_result = _handle_evaluation_errors(result, op_str, left_value, right_value)
        if not success:
            result = None
        else:
            result = eval_result
    finally:
        if is_reversed:
            left_op, right_op = right_op, left_op

    if _TEST_MODE:
        _store_test_result(result is not None)

    if result is None:
        result = _evaluate_standard(op, op_str, left_op, right_op)

    return result
# CCN: 6, NLOC: ~20, Parameters: 4