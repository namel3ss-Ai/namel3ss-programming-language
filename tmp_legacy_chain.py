import textwrap
from namel3ss.parser.program import LegacyProgramParser

source = textwrap.dedent("""
    app:
        name: test_app

    define chain "support_chain":
        workflow:
            - step "greet":
                template "greet_template"
            - step "respond":
                llm "chat_model" context greet
    """).strip()
legacy = LegacyProgramParser(source)
module = legacy.parse()
print('chains', len(module.body[0].chains))
