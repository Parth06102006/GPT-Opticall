
import json
import re

def _extract_json(text: str) -> dict:
    if not text:
        return {}

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
    
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    repaired = cleaned
    if repaired.count('"') % 2 != 0:
        repaired += '"'
        
    stack = []
    for char in repaired:
        if char == '{':
            stack.append('}')
        elif char == '[':
            stack.append(']')
        elif char == '}':
            if stack and stack[-1] == '}':
                stack.pop()
        elif char == ']':
            if stack and stack[-1] == ']':
                stack.pop()
    
    if stack:
        repaired += "".join(reversed(stack))
            
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    try:
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_part = cleaned[start_idx : end_idx + 1]
            return json.loads(json_part)
    except json.JSONDecodeError:
        pass

    raise ValueError("Could not parse JSON")

# Test cases
tests = [
    ('{"key": "value"}', {"key": "value"}),
    ('```json\n{"key": "value"}\n```', {"key": "value"}),
    ('{"segments": [{"text": "Hello"', {"segments": [{"text": "Hello"}]}),
    ('{"segments": [{"text": "H', {"segments": [{"text": "H"}]}),
    ('{"key": "val', {"key": "val"}),
]

for i, (input_str, expected) in enumerate(tests):
    try:
        result = _extract_json(input_str)
        assert result == expected
        print(f"Test {i} passed")
    except Exception as e:
        print(f"Test {i} failed: {e}")
        print(f"  Input: {input_str}")
        print(f"  Result: {result if 'result' in locals() else 'N/A'}")

print("All tests completed.")
