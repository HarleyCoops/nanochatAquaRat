"""
Test the _extract_letter function with various output formats
to identify which patterns it can and cannot match.
"""
from tasks.aqua import _extract_letter

# Test cases: (output_text, expected_letter, description)
test_cases = [
    # Standard formats that should work
    ("Answer: A", "A", "Standard 'Answer: X' format"),
    ("Answer: B", "B", "Standard with different letter"),
    ("Answer:C", "C", "No space after colon"),
    ("Answer - D", "D", "Dash instead of colon"),
    ("The answer is E", "E", "Bare letter fallback"),
    
    # Formats that might be causing issues
    ("Answer: A)", "A", "Answer with closing paren"),
    ("Answer: B.", "B", "Answer with period"),
    ("Answer: C\n", "C", "Answer with newline"),
    ("Answer: (D)", "D", "Answer with parens around letter"),
    ("Answer: The correct option is E", "E", "Verbose format"),
    
    # Edge cases
    ("The correct answer is option A)", "A", "Option A with paren"),
    ("A) is the right answer", "A", "Letter at start"),
    ("I choose option D)", "D", "Choose format with paren"),
    ("Based on the calculation, the answer is B.", "B", "Full sentence"),
    
    # Problematic formats
    ("Answer: A) explanation here", "A", "Answer with explanation"),
    ("Answer: option A", "A", "Word 'option' before letter"),
    ("The answer should be choice C)", "C", "Choice format"),
    ("D) 21.5 is correct", "D", "Letter with value"),
]

print("Testing _extract_letter function:")
print("=" * 80)

passed = 0
failed = 0
failed_cases = []

for text, expected, description in test_cases:
    result = _extract_letter(text)
    status = "✓" if result == expected else "✗"
    
    if result == expected:
        passed += 1
    else:
        failed += 1
        failed_cases.append((text, expected, result, description))
    
    print(f"{status} {description}")
    print(f"  Input: {repr(text)}")
    print(f"  Expected: {expected}, Got: {result}")
    print()

print("=" * 80)
print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")

if failed_cases:
    print("\nFailed cases that need fixing:")
    print("-" * 80)
    for text, expected, result, description in failed_cases:
        print(f"Description: {description}")
        print(f"Input: {repr(text)}")
        print(f"Expected: {expected}, Got: {result}")
        print()
