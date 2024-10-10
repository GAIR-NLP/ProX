def split_doc(text):
    """
    suppose text is already wrapped in [xxx]some text 1\n[xxx]some text2 ...
    split by line and keep only pure line text, without pre-pended line number
    """
    lines = text.split("\n")
    processed_lines = []
    for line in lines:
        processed_line = line[:]
        processed_lines.append(processed_line)
    return processed_lines


def execute_operations(text, operations):
    """
    Executes a series of operations on the text and collects noisy and clean string pairs.

    :param text: str, the input text wrapped in [xxx]some text 1\n[xxx]some text2 ...
    :param operations: list of str, operations to be performed on the text
    :return: tuple(str, list of tuples), the modified text and list of (noisy, clean) string pairs
    """
    # Split the text into lines
    lines = split_doc(text)

    # Set to store line numbers to be removed
    lines_to_remove = set()

    # List to store (noisy, clean) string pairs
    normalization_pairs = []

    # Parse the operations
    intervals = []

    num_error_op = 0
    for operation in operations:
        if operation.startswith("remove_lines") and "all=True" not in operation:
            # Extract the start and end line numbers
            try:
                start_idx = operation.index("(") + 1
            except:
                num_error_op += 1
                continue
            try:
                end_idx = operation.index(")")
            except:
                num_error_op += 1
                end_idx = -1

            params = operation[start_idx:end_idx].split(",")
            try:
                start = int(params[0].split("=")[1])
                end = int(params[1].split("=")[1])
            except:
                num_error_op += 1
                continue
            intervals.append((start, end))

            # Add the specified lines (1-based indexing) to the set
            for line_num in range(start, end + 1):
                lines_to_remove.add(line_num)  # Convert to 0-based indexing

        elif operation.startswith("normalize"):
            # Extract the strings to be normalized
            try:
                start_idx = operation.index("(") + 1
            except:
                num_error_op += 1
                continue
            try:
                end_idx = operation.index(")")
            except:
                num_error_op += 1
                end_idx = -1
            params = operation[start_idx:end_idx]
            params = params.split(",")

            if len(params) == 1:
                # Remove the specified string
                try:
                    source_str = params[0].split("=")[1].strip('"')
                    text = text.replace(source_str, "")
                    normalization_pairs.append((source_str, ""))
                    # print(normalization_pairs)
                except:
                    num_error_op += 1
                    continue
            elif len(params) == 2:
                # Replace source string with target string
                try:
                    source_str = params[0].split("=")[1].strip('"')
                    target_str = params[1].split("=")[1].strip('"')
                except:
                    num_error_op += 1
                    continue
                if source_str != " " and source_str != "":
                    text = text.replace(source_str, target_str)
                normalization_pairs.append((source_str, target_str))

        elif operation.startswith("remove_lines") and "all=True" in operation:
            # Remove all lines
            lines_to_remove = set(range(len(lines)))
    # Remove the specified lines after normalization to ensure correct line numbers
    lines = split_doc(text)

    processed_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]
    noise_lines = [line for i, line in enumerate(lines) if i in lines_to_remove]

    intervals = merge_intervals(intervals)
    remove_operations = []
    for interval in intervals:
        if interval[0] >= len(lines):
            num_error_op += 1
            continue
        elif interval[1] >= len(lines):
            num_error_op += 1
            interval = [interval[0], len(lines) - 1]
        remove_operations.append([interval[0], interval[1]])

    # clean up cache
    del lines, lines_to_remove, intervals, text

    return (
        "\n".join(processed_lines),
        normalization_pairs,
        noise_lines,
        remove_operations,
        num_error_op,
    )


def merge_intervals(intervals):
    if not intervals:
        return []

    # Step 1: Sort the intervals by the start point
    intervals.sort(key=lambda x: x[0])

    # Step 2: Initialize the list to hold the merged intervals
    merged = [intervals[0]]

    for current in intervals[1:]:
        previous = merged[-1]
        # Step 3: If current interval overlaps with the previous, merge them
        if current[0] <= previous[1] + 1:
            merged[-1] = (previous[0], max(previous[1], current[1]))
        else:
            # Otherwise, add the current interval to the list
            merged.append(current)

    return merged


def execute_meta_operations(
    text, operations, threshold_1=0.0, threshold_2=0.95, error_op=2
):
    program = operations.strip(" ").split("\n")
    (
        processed_text,
        normalization_pairs,
        noise_lines,
        remove_operations,
        num_error_op,
    ) = execute_operations(text, program)
    REMOVE_THRESHOLD = threshold_1
    ERROR_OP = error_op

    del noise_lines, remove_operations, normalization_pairs

    len_processed_text = len(processed_text.split())
    len_text = len(text.split())

    # return unqualified text
    if (
        len_processed_text / len_text * 1.0 <= (1 - threshold_2)
        and num_error_op < ERROR_OP
    ):
        return ""
    elif len_processed_text <= 10:  # @fan add-hoc rule integrated from the fineweb
        return ""
    # return processed_text
    elif (
        len_processed_text / len_text * 1.0 >= REMOVE_THRESHOLD
        and num_error_op < ERROR_OP
    ):
        return processed_text
    # @fan too many error operations then we ignore the program
    else:
        return text
