

def concat_conversation(lines:list):
    conversation = []
    for line in lines:
        role = line["role"]
        words = line["words"].strip()
        conversation.append(f"{role}:{words}")
    return "\n".join(conversation)