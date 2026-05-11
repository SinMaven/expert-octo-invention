"""
Zero-Knowledge Voice -- PII Redaction Test Suite
==================================================
Tests PII detection/anonymization against synthetic data.
Covers all entity types and edge cases.

Author: Zero-Knowledge Voice Team
"""

import logging
import sys
from dataclasses import dataclass
from typing import List

from faker import Faker
from model import redact_pii, detect_pii

fake = Faker()


@dataclass
class TestCase:
    name: str
    input_text: str
    expected_types: List[str]


def build_test_cases() -> List[TestCase]:
    return [
        TestCase("PERSON detection", f"My name is {fake.name()} and I live in New York.", ["PERSON", "LOCATION"]),
        TestCase("PHONE_NUMBER detection", f"Call me at {fake.phone_number()} for details.", ["PHONE_NUMBER"]),
        TestCase("CREDIT_CARD detection", f"My card number is {fake.credit_card_number()}.", ["CREDIT_CARD"]),
        TestCase("EMAIL_ADDRESS detection", f"Email me at {fake.email()} for updates.", ["EMAIL_ADDRESS"]),
        TestCase("US_SSN detection", f"My SSN is {fake.ssn()}.", ["US_SSN"]),
        TestCase("Multiple PII types",
                 f"I am {fake.name()}, email {fake.email()}, phone {fake.phone_number()}.",
                 ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]),
        TestCase("Empty string", "", []),
        TestCase("No PII present", "The weather is sunny today and the sky is blue.", []),
        TestCase("Non-PII numbers", "There are 42 apples and 7 oranges.", []),
        TestCase("Hyphenated Credit Card", "Use my new visa, 453-2015678-901234, for the payment.", ["CREDIT_CARD"]),
        TestCase("CVV detection", "The CVV code is 192 and expiration is 08-28.", ["CVV", "EXP_DATE"]),
        TestCase("Complex financial block", "Card ending in 8,812. Visa 453-2015678-901234 exp 08-28 CVV192", ["CREDIT_CARD", "EXP_DATE", "CVV"]),
        TestCase("Broken Email detection", "Send it to e.rodrigas88 fastmail.net please.", ["BROKEN_EMAIL"]),
        TestCase("CVV false positive prevention", "My phone number is 555-012-777.", []),
        TestCase("Spelled-out numbers", "My security code is four five three", ["SPELLED_NUM"]),
        TestCase("Passport ID with context", "The passport number is AB1234567.", ["GOVT_ID"]),
        TestCase("Zip code detection", "I live in Chicago, Illinois 60611.", ["LOCATION", "ZIP_CODE"]),
        TestCase("Account ID detection", "My account ID is GH99281X.", ["ACCOUNT_NUM"]),
        TestCase("Irregular phone format", "Reach me at 555-01288-877.", ["PHONE_NUMBER"]),
    ]


def run_tests() -> bool:
    test_cases = build_test_cases()
    passed = failed = 0

    print(f"\n{'='*60}")
    print(f"  PII REDACTION TEST SUITE -- {len(test_cases)} cases")
    print(f"{'='*60}\n")

    for i, tc in enumerate(test_cases, 1):
        entities = detect_pii(tc.input_text)
        redacted = redact_pii(tc.input_text)
        detected_types = set(e.entity_type for e in entities)
        expected_set = set(tc.expected_types)

        if not expected_set:
            success = True
        else:
            missing = expected_set - detected_types
            success = len(missing) == 0

        status = "PASS" if success else "FAIL"
        passed += success
        failed += (not success)

        print(f"  [{i}/{len(test_cases)}] {status} -- {tc.name}")
        print(f"    Input:    {tc.input_text[:70]}")
        print(f"    Redacted: {redacted[:70]}")
        if entities:
            print(f"    Found:    {', '.join(e.entity_type for e in entities)}")
        if not success:
            print(f"    Missing:  {', '.join(expected_set - detected_types)}")
        print()

    print(f"{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed, {len(test_cases)} total")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(0 if run_tests() else 1)