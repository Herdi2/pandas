def coverage_wrapper(branch_coverage):
    return lambda x=branch_coverage : finalize_coverage(x)

def finalize_coverage(branch_coverage):
    """After all test cases execute, compute and save the coverage report."""
    total_branches = max(branch_coverage.keys(), default=0)  # Prevent KeyError if empty
    executed_branches = len([b for b, count in branch_coverage.items() if count > 0])
    coverage_score = (executed_branches / total_branches) * 100 if total_branches > 0 else 0
    for branch, count in sorted(branch_coverage.items()):
        print(f"Branch {branch}: executed {count} times\n")
    print(f"Manual Coverage Score: {coverage_score:.2f}%")
    
    print("Final Manual Coverage Report Generated!")