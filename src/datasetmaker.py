def is_speaker_line(index: int, lines: list[str]) -> bool:
    """
    Checks if a line at a given index is a speaker's name.
    A line is considered a speaker's name if the next line is '•'.
    """
    # Check bounds and the '•' pattern
    if index + 1 < len(lines) and lines[index + 1].strip() == '•':
        # Exclude metadata blocks that follow the same pattern
        if lines[index].strip() not in ["AI SUMMARY", "AI SENTIMENT"]:
            return True
    return False


def clean_conversation_multiline(raw_text: str) -> str:
    """
    Parses conversation text with dynamic names and multi-line messages.

    It identifies the customer as the first speaker and correctly groups
    all subsequent lines as part of a single message until a new speaker
    is found.

    Args:
        raw_text: A string containing the entire raw conversation.

    Returns:
        A formatted string of the conversation.
    """
    lines = raw_text.strip().split('\n')

    # 1. Dynamically find the customer's name (the first speaker)
    customer_name = None
    for i, line in enumerate(lines):
        if is_speaker_line(i, lines):
            customer_name = line.strip()
            break

    if not customer_name:
        return "Could not identify any speakers in the conversation."

    # Define lines/words to ignore completely
    ignore_keywords = {"attachment", "Урьдчилан үзэх", "Delivered", "Sent", "Seen"}

    formatted_conversation = []

    # Use a while loop to manually control the index
    i = 0
    while i < len(lines):
        # 2. Check if the current line is the start of a message block
        if is_speaker_line(i, lines):
            speaker = lines[i].strip()

            # 3. Capture all lines belonging to this message
            message_parts = []
            # The message content starts 3 lines down from the speaker's name
            message_start_index = i + 3

            j = message_start_index
            while j < len(lines):
                # Stop when we find the next speaker
                if is_speaker_line(j, lines):
                    break

                line_content = lines[j].strip()

                # Skip ignored keywords and empty lines
                if line_content and line_content not in ignore_keywords:
                    message_parts.append(line_content)

                j += 1  # Move to the next line

            # 4. Join, format, and append the complete message
            if message_parts:
                # Join the parts into a single message string with newlines
                full_message = "\\n".join(message_parts)

                # Assign the correct label ('customer' or 'customer service')
                if speaker == customer_name:
                    speaker_label = f"'{speaker}'"
                else:
                    speaker_label = "'customer service'"

                formatted_conversation.append(f"{speaker_label}: \"{full_message}\"")

            # Jump the main loop index to where the last message ended
            i = j
        else:
            # If it's not a speaker line, just advance to the next
            i += 1

    return "\n".join(formatted_conversation)


# The new raw conversation text you provided
new_raw_conversation = """
Б. Цэрэн Надмид
•
2025-10-02 17:06
Get started
Ger Internet
•
2025-10-02 17:06
Сайн байна уу? Гэр интернэтийн 24/7 онлайн туслах U-Bot байна. Та манай үйлчилгээтэй холбоотой бүх төрлийн асуултаа надаас асуугаарай 😊

Б. Цэрэн Надмид
•
2025-10-02 17:06
Ажилтантай холбогдох

Б. Цэрэн Надмид
•
2025-10-02 17:08
Nuuts kod shinechleh
batgerel.g
•
2025-10-02 17:14
Сайн байна уу? Би Аялгуу байна. 🙋‍♀️  Гэр дугаар болон нэрээ илгээгээрэй.
Seen
"""

# Run the improved script and print the full, correct output
cleaned_output = clean_conversation_multiline(new_raw_conversation)
print(cleaned_output)
