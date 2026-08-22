from app.output_governance.pii_scanner import scan


def test_clean_text_passes_through_unchanged():
    text = "The weather today is sunny with a light breeze."

    result = scan(text)

    assert result.found == []
    assert result.redacted_text == text


def test_empty_text_returns_empty_result():
    result = scan("")

    assert result.found == []
    assert result.redacted_text == ""


def test_person_name_is_caught():
    result = scan("Please forward this to Sarah Chen for review.")

    types = [m.type for m in result.found]
    assert "PERSON" in types
    assert "[REDACTED:PERSON]" in result.redacted_text
    assert "Sarah Chen" not in result.redacted_text


def test_location_gpe_is_caught():
    result = scan("The weather in Paris is lovely this time of year.")

    types = [m.type for m in result.found]
    assert "GPE" in types
    assert "Paris" not in result.redacted_text


def test_organization_is_caught():
    result = scan("She previously worked at Microsoft before joining the team.")

    types = [m.type for m in result.found]
    assert "ORG" in types
    assert "Microsoft" not in result.redacted_text


def test_email_is_caught():
    result = scan("You can reach me at jane.doe@example.com anytime.")

    types = [m.type for m in result.found]
    assert "EMAIL" in types
    assert "jane.doe@example.com" not in result.redacted_text


def test_phone_number_is_caught():
    result = scan("Call the office at 415-555-0142 during business hours.")

    types = [m.type for m in result.found]
    assert "PHONE" in types
    assert "415-555-0142" not in result.redacted_text


def test_credit_card_shaped_number_is_caught():
    result = scan("The card number on file is 4111 1111 1111 1111 for this account.")

    types = [m.type for m in result.found]
    assert "CREDIT_CARD" in types
    assert "4111 1111 1111 1111" not in result.redacted_text


def test_anthropic_api_key_is_caught():
    fake_key = "sk-ant-api03-" + "a" * 40
    result = scan(f"Here is the key: {fake_key} — keep it secret.")

    types = [m.type for m in result.found]
    assert "API_KEY" in types
    assert fake_key not in result.redacted_text


def test_google_ai_studio_key_is_caught():
    fake_key = "AQ." + "B" * 40
    result = scan(f"My key is {fake_key} for this project.")

    types = [m.type for m in result.found]
    assert "API_KEY" in types
    assert fake_key not in result.redacted_text


def test_github_token_is_caught():
    fake_token = "ghp_" + "c" * 36
    result = scan(f"Use this token: {fake_token} in CI.")

    assert "API_KEY" in [m.type for m in result.found]
    assert fake_token not in result.redacted_text


def test_redaction_preserves_surrounding_sentence_structure():
    text = "Contact Sarah Chen at sarah.chen@example.com or call 415-555-0142."

    result = scan(text)

    assert result.redacted_text.startswith("Contact ")
    assert result.redacted_text.endswith(".")
    assert " at " in result.redacted_text
    assert " or call " in result.redacted_text


def test_multiple_distinct_pii_types_all_caught_in_one_pass():
    text = "John works at Google in Mountain View and his email is john@example.com."

    result = scan(text)

    types = {m.type for m in result.found}
    assert "PERSON" in types
    assert "ORG" in types
    assert "GPE" in types
    assert "EMAIL" in types


def test_spans_are_reported_relative_to_original_text():
    text = "Email me at test@example.com please."

    result = scan(text)

    email_match = next(m for m in result.found if m.type == "EMAIL")
    start, end = email_match.span
    assert text[start:end] == "test@example.com"


def test_overlapping_matches_do_not_produce_double_redaction():
    # The API-key regex match should win over any spurious shorter NER
    # hit inside the same span, not produce two overlapping [REDACTED:]
    # markers.
    fake_key = "sk-ant-api03-" + "x" * 40
    result = scan(f"key: {fake_key}")

    assert result.redacted_text.count("[REDACTED:") == 1
