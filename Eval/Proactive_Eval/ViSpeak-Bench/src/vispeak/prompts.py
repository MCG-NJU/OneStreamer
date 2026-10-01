ACTIVE_TASKS = (
    "Gesture_Understanding",
    "Anomaly_Warning",
    "Humor_Reaction",
    "Visual_Interruption",
    "Visual_Termination",
    "Visual_Wake-Up",
)


TASK_PROMPTS = {
    "Anomaly_Warning": (
        "Continuously monitor the video for a clearly unusual or dangerous "
        "event. Once there is enough visual evidence, briefly describe what "
        "happened and give safe, practical advice appropriate to the event."
    ),
    "Gesture_Understanding": (
        "Continue observing the person in the video for a meaningful gesture. "
        "Once the gesture is clear, identify it and respond naturally according "
        "to the preceding conversation."
    ),
    "Humor_Reaction": (
        "Continuously watch for a clearly humorous visual event. Once the event "
        "is clear enough to understand, react naturally and briefly explain what "
        "makes it funny."
    ),
    "Visual_Interruption": (
        "Treat the assistant response in the conversation context as ongoing. "
        "Continue watching the user for a clear body-language signal asking the "
        "assistant to stop or be interrupted. Once that signal is clear, stop "
        "and briefly acknowledge the interruption."
    ),
    "Visual_Termination": (
        "Continue watching the user after the conversation. If the user makes a "
        "clear goodbye or conversation-ending gesture, respond naturally with a "
        "brief closing message."
    ),
    "Visual_Wake-Up": (
        "Continuously watch for a greeting or wake-up gesture directed at you. "
        "Once the gesture is clear, respond naturally with a brief greeting and "
        "an offer to help."
    ),
}


CONTEXT_TASKS = {
    "Gesture_Understanding",
    "Visual_Interruption",
    "Visual_Termination",
}


def _conversation_context(sample):
    lines = []
    for turn in sample.get("conversations", [])[:-1]:
        role = turn.get("from", "unknown")
        if role == "human":
            label = "User"
        elif role == "gpt":
            label = "Assistant"
        else:
            label = role.capitalize()
        value = str(turn.get("value", "")).strip()
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def build_online_task_prompt(task_name, sample):
    if task_name not in TASK_PROMPTS:
        raise KeyError(f"Unsupported online ViSpeak task: {task_name}")

    task_prompt = TASK_PROMPTS[task_name]
    if task_name not in CONTEXT_TASKS:
        return task_prompt

    context = _conversation_context(sample)
    if not context:
        return task_prompt
    return f"Conversation context:\n{context}\n\nStreaming task:\n{task_prompt}"
