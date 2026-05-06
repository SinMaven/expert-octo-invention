from faker import Faker
from model import redact_pii

fake = Faker()

def test_pii_redaction():
    test_cases = [
        f"My name is {fake.name()} and I live in New York",
        f"Call me at {fake.phone_number()}",
        f"My card number is {fake.credit_card_number()}",
        f"Email me at {fake.email()}",
        f"My SSN is {fake.ssn()}"
    ]

    print("=" * 50)
    print("PII Redaction Test")
    print("=" * 50)

    for text in test_cases:
        redacted = redact_pii(text)
        print(f"Original : {text}")
        print(f"Redacted : {redacted}")
        print("-" * 50)

if __name__ == "__main__":
    test_pii_redaction()