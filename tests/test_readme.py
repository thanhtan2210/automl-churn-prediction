from scripts.render_readme import README, blocks, render


def test_readme_numbers_match_the_reports():
    text = README.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert render(text) == text, "README.md is out of date: run python -m scripts.render_readme"


def test_every_block_is_present():
    text = README.read_text(encoding="utf-8")
    for name in blocks():
        assert f"<!-- {name}:start -->" in text and f"<!-- {name}:end -->" in text
