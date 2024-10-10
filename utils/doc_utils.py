def execute_meta_operations(text, operations):
    program = operations.strip(" ")
    if "drop" in program:
        return ""
    else:
        return text
