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
                    speaker_label = f"'{speaker}'"

                formatted_conversation.append(f"{speaker_label}: \"{full_message}\"")

            # Jump the main loop index to where the last message ended
            i = j
        else:
            # If it's not a speaker line, just advance to the next
            i += 1

    return "\n".join(formatted_conversation)


# The new raw conversation text you provided
new_raw_conversation = """
Zooey Ulziibayar
•
2025-10-11 19:29
Ажилтантай холбогдох

Zooey Ulziibayar
•
2025-10-11 19:30
E-mongolia deer zeeliin medeelel der unitel turees gesen turliin zeel baina, shalgaad ugch boloh u, bi say l medle. Odoo bas unitel dugaargui, gadaadad baidag umaa
erdenechimeg.mun
•
2025-10-11 19:30
Сайн байна уу? Би Чимгээ байна 🙋‍♀️ Овог нэрээ болон рд илгээгээрэй

Zooey Ulziibayar
•
2025-10-11 19:30
Burenzaya Ulziibayar, USh96052509
erdenechimeg.mun
•
2025-10-11 19:31
УШ юмуу

Zooey Ulziibayar
•
2025-10-11 19:31
УШ96052509
erdenechimeg.mun
•
2025-10-11 19:33
Түр хүлээгээрэй
erdenechimeg.mun
•
2025-10-11 20:18
Хүлээцтэй хандсан танд маш их баярлалаа 🤗 407030132 гэрээтэй Юнивишнийн үйлчилгээ харагдаж байна.
erdenechimeg.mun
•
2025-10-11 20:18
Та Юнивишнийн мэдээллийг дараах холбоосоор чат бот руу хандан ажилтантай холбогдон аваарай 😊🔗
Seen
"""

# Run the improved script and print the full, correct output
cleaned_output = clean_conversation_multiline(new_raw_conversation)
print(cleaned_output)
