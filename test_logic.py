"""Quick smoke test for NLP pipeline."""
from question_generator import generate_mcqs, generate_quiz_from_topics
from pdf_extractor import detect_topics, clean_text

sample = (
    "Computer Networks is the study of communication between computers. "
    "A protocol is a set of rules that governs data communication between devices. "
    "TCP IP is the foundational protocol suite used by the Internet. "
    "The OSI model defines seven layers of network communication. "
    "A router is a networking device that forwards data packets between networks. "
    "DNS stands for Domain Name System and translates domain names into IP addresses. "
    "HTTP is the Hypertext Transfer Protocol used for transmitting web pages. "
    "Encryption is the process of converting data into a coded format. "
    "A firewall is a network security device that monitors incoming and outgoing traffic. "
    "Bandwidth refers to the maximum rate of data transfer across a network path. "
    "A switch is a device that connects devices on a computer network. "
    "The IP address is a numerical label assigned to each device on a network. "
    "Latency is the time it takes for data to travel from source to destination. "
    "A hub is a basic networking device that connects multiple computers together. "
    "Subnetting is the practice of dividing a network into two or more subnetworks. "
)

# Test topic detection
topics = detect_topics(sample)
print(f"Topics detected: {list(topics.keys())}")

# Test MCQ generation
qs = generate_mcqs(sample, topic="Computer Networks", num_questions=5)
print(f"\nGenerated {len(qs)} questions:")
for i, q in enumerate(qs, 1):
    print(f"\n  Q{i}: {q['question']}")
    for label, text in q['options'].items():
        marker = " <-- ANSWER" if label == q['answer'] else ""
        print(f"       {label}. {text}{marker}")
    print(f"       Topic: {q['topic']}")

assert len(qs) > 0, "No questions generated!"
print("\n✅ All tests passed — app is ready.")
