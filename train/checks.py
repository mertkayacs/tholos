"""Training-only checks for material facts in successful tool calls.

These checks accept reordered prose and common count/comparison wording. They
are deliberately lexical: arbitrary synonyms, implicit comparisons, and general
natural-language entailment are outside their scope. Identifiers retain exact
word boundaries, so a fact hidden inside an unrelated identifier does not count.
"""

import json
import re
from decimal import Decimal, InvalidOperation

from tholos.bench.runner import matches

_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty",
]
_NUMBERS = {word: Decimal(i) for i, word in enumerate(_WORDS)}
_NUMBERS.update({f"twenty {word}": Decimal(20 + i) for i, word in enumerate(_WORDS[1:5], 1)})
_NUMBER_RE = re.compile(
    r"(?<![\w.])(?:[+-]?\d+(?:\.\d+)?|twenty[ -](?:one|two|three|four)|"
    + "|".join(reversed(_WORDS)) + r")(?!\w|\.\d)",
    re.IGNORECASE,
)
_ROW_RE = re.compile(r"^(?:\s+\w+){0,3}\s+(?:rows?|entries|entry|records?|items?)\b")
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "to", "for", "from", "in", "on", "at", "as",
    "with", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this",
    "that", "these", "those", "has", "have", "had", "owner", "asked", "request",
    "please", "pls", "needs", "need", "wants", "want", "should", "would", "could",
    "can", "handle", "look", "into", "review", "record", "log", "noted", "note",
    "says", "said",
}


def _token_pattern(value):
    value = str(value).strip().casefold()
    pattern = re.escape(value)
    if value and (value[0].isalnum() or value[0] == "_"):
        pattern = r"(?<!\w)" + pattern
    if value and (value[-1].isalnum() or value[-1] == "_"):
        pattern += r"(?!\w)"
    return pattern


def contains(text, value):
    """Match an exact material token or phrase, without identifier substrings."""
    return (bool(str(value).strip())
            and re.search(_token_pattern(value), text.casefold()) is not None)


def _numbers(text):
    for match in _NUMBER_RE.finditer(text):
        token = match.group().casefold()
        word = token.replace("-", " ")
        yield match, _NUMBERS[word] if word in _NUMBERS else Decimal(token)


def facts(text, wanted):
    """Match factual words without requiring their original sentence order."""
    text = text.casefold()
    if not _valid_statement(text):
        return False
    for fact in wanted:
        words = re.findall(r"[+-]\d+(?:\.\d+)?|\w+(?:[-.]\w+)*",
                           str(fact).casefold())
        material = [word for word in words if word not in _STOP_WORDS]
        if not material:
            material = words
        for word in material:
            if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", word):
                if not contains(text, word) and Decimal(word) not in {
                    value for _, value in _numbers(text)
                }:
                    return False
            elif word in {"true", "false"}:
                aliases = ("true", "yes") if word == "true" else ("false", "no")
                if not any(contains(text, alias) for alias in aliases):
                    return False
            elif word in _NUMBERS:
                if (not contains(text, word)
                        and _NUMBERS[word] not in {value for _, value in _numbers(text)}):
                    return False
            elif not contains(text, word):
                return False
    return True


def _calls(messages):
    """Read persisted assistant steps and the immediately following result.

    A successful finish ends the transcript without a tool response. Other calls
    require a result, excluding errors and unexecuted approval requests.
    """
    calls = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        try:
            step = json.loads(message.get("content", ""))
        except (TypeError, ValueError):
            continue
        if not isinstance(step, dict) or not isinstance(step.get("args"), dict):
            continue
        following = messages[index + 1] if index + 1 < len(messages) else None
        result = None
        if following and following.get("role") == "user":
            content = following.get("content", "")
            if isinstance(content, str):
                match = re.fullmatch(
                    r"\s*<tool_response>\s*(.*?)\s*</tool_response>\s*", content, re.S,
                )
                if match:
                    try:
                        result = json.loads(match.group(1))
                    except ValueError:
                        continue
        if result is None:
            if step.get("tool") != "finish" or following is not None:
                continue
        elif not isinstance(result, dict) or "error" in result or result.get("waiting"):
            continue
        calls.append(step | {"_result": result})
    return calls


def _texts(calls, check):
    tool = check.get("tool", "finish")
    target = check.get("target", {})
    field = {"finish": "summary", "note_write": "text", "remember": "fact"}[tool]
    return [str(call["args"].get(field, "")) for call in calls
            if call.get("tool") == tool
            and all(call["args"].get(key) == value for key, value in target.items())]


def _note_texts(scenario, calls, check):
    notes = {note["title"]: note["body"]
             for note in scenario.get("workspace", {}).get("notes", [])}
    edited = set()
    interference = scenario.get("interfere", {})
    interfered = False
    for call in calls:
        if call.get("tool") == "note_write":
            args = call["args"]
            title = args["title"]
            text = str(args.get("text", ""))
            notes[title] = notes.get(title, "") + text if args.get("mode") == "append" else text
            edited.add(title)
        if (not interfered and call.get("tool") == interference.get("after_tool")
                and call["args"].get("title") == interference.get("note")):
            if interference.get("note"):
                title = interference["note"]
                notes[title] = notes.get(title, "") + interference.get("append", "")
            interfered = True
    target = check.get("target", {})
    return [text for title, text in notes.items()
            if title in edited and ("title" not in target or title == target["title"])]


def _state(scenario, calls, check):
    tables = {}
    next_id = 1
    for table in scenario.get("workspace", {}).get("tables", []):
        rows = {}
        for row in table.get("rows", []):
            rows[next_id] = {column: row.get(column, "") for column in table["columns"]}
            next_id += 1
        tables[table["name"]] = {"columns": table["columns"], "rows": rows, "proven": True}
    interference = scenario.get("interfere", {})
    interfered = False
    for call in calls:
        tool, args, result = call.get("tool"), call["args"], call["_result"]
        name = args.get("table")
        if tool == "table_create":
            tables[name] = {"columns": args["columns"], "rows": {},
                            "proven": bool(result and "id" in result
                                           and not result.get("truncated"))}
        elif tool in {"table_add", "table_update"} and name in tables:
            table = tables[name]
            if tool == "table_add":
                ids = result.get("rows")
                if (not isinstance(ids, list) or len(ids) != len(args.get("rows", []))
                        or any(type(id) is not int for id in ids) or result.get("truncated")):
                    table["proven"] = False
                else:
                    for id, row in zip(ids, args["rows"], strict=True):
                        table["rows"][id] = {
                            column: row.get(column) if row.get(column) is not None else ""
                            for column in table["columns"]
                        }
            else:
                row = result.get("row")
                if result.get("truncated") and args.get("row") in table["rows"]:
                    table["rows"][args["row"]].update({
                        key: value for key, value in args.get("values", {}).items()
                        if value is not None
                    })
                elif (not isinstance(row, dict) or row.get("id") != args.get("row")
                        or not isinstance(row.get("data"), dict)
                        or set(row["data"]) != set(table["columns"])):
                    table["proven"] = False
                else:
                    table["rows"][row["id"]] = row["data"]
        if (not interfered and tool == interference.get("after_tool")
                and name == interference.get("table") and name in tables):
            for row in tables[name]["rows"].values():
                if all(matches(row.get(key), value)
                       for key, value in interference.get("row_match", {}).items()):
                    row.update(interference.get("set", {}))
                    interfered = True
                    break
    table = tables.get(check["table"])
    if (not table or not table["proven"] or set(table["columns"]) != set(check["columns"])
            or len(table["rows"]) != len(check["rows"])):
        return False
    remaining = list(table["rows"].values())
    for expected in check["rows"]:
        for index, actual in enumerate(remaining):
            if all((actual.get(column) in (None, "") if expected.get(column) in (None, "")
                    else _state_value(actual.get(column), expected.get(column)))
                   for column in check["columns"]):
                remaining.pop(index)
                break
        else:
            return False
    return True


def _state_value(actual, expected):
    if type(expected) is bool:
        return type(actual) is bool and actual == expected or (
            isinstance(actual, str) and actual.strip().casefold() == str(expected).casefold()
        )
    return matches(actual, expected)


def _count(text, check):
    if not _valid_statement(text.casefold()):
        return False
    if check.get("context") and not contains(text, check["context"]):
        return False
    wanted = Decimal(str(check["value"]))
    numbers = list(_numbers(text))
    if wanted not in {value for _, value in numbers}:
        return False
    explicit = []
    current = []
    historical = []
    for match, value in numbers:
        tail = text[match.end():].casefold()
        before = re.split(r"[.;!?\n]", text[:match.start()])[-1][-45:].casefold()
        old = bool(re.search(
            r"\b(?:previously|previous|prior|old|earlier|historical|yesterday|last)\b", before,
        ))
        label = bool(re.search(
            r"\b(?:count|total)\b\s*(?:(?:is|are|of)\s*|[:=]\s*)?$", before,
        ))
        if (re.search(r"\b(?:now|currently|actually|current|actual|today)\b"
                      r"(?:\s+(?:count|total|is|are|has|there|of)|\s*[:=]){0,3}\s*$", before)
                or re.match(r"\s+(?:is|are)\s+(?:the\s+)?(?:current|actual)\b", tail)
                or (label and not old)):
            current.append(value)
        if old:
            historical.append(value)
        if _ROW_RE.match(tail) or label:
            # A denominator does not contradict the number of matching rows.
            if re.search(r"(?:out of|of)\s*$", before):
                continue
            explicit.append((value, old))
    if current and any(value != wanted for value in current):
        return False
    if wanted in historical and not current:
        return False
    return not explicit or all(value == wanted or (old and current) for value, old in explicit)


def _valid_statement(text):
    return not re.search(
        r"^\s*(?:incorrect|wrong|retracted)\s*:"
        r"|\b(?:reports?|facts?|values?|information)\s+"
        r"(?:are|were|is|was|have been)\s+(?:incorrect|wrong|retracted)\b"
        r"|\b(?:does not apply|do not use|disregard this|ignore this)\b", text,
    )


def _direction(text, first, second):
    """Does the sentence explicitly say that first outranks second?"""
    a, b = re.escape(first), re.escape(second)
    patterns = [
        rf"{a}\s*>\s*{b}",
        rf"{b}\s*<\s*{a}",
        rf"{a}[^.;!?\n]{{0,100}}\b(?:more|most|higher|greater|larger|wins?|winner|outnumbers?)\b[^.;!?\n]{{0,80}}{b}",
        rf"{b}[^.;!?\n]{{0,100}}\b(?:less|fewer|lower|smaller)\b[^.;!?\n]{{0,80}}{a}",
        rf"\bmore\s+{a}[^.;!?\n]{{0,60}}\bthan\s+{b}",
        rf"\b(?:winner|answer|more common|most common)\s*(?:is|:)?\s*{a}\b",
        rf"\b(?:so|therefore|thus)\s+(?:the\s+(?:winner|answer)\s+is\s+)?{a}\b",
        rf"{a}\s+(?:(?:is|has|appears|occurs)\s+)?(?:the\s+)?"
        r"(?:winner|more common|most common|more rows|higher count|wins)\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _comparison(text, check):
    winner, loser = str(check["winner"]), str(check["loser"])
    aliases = {"true": ["true", "yes"], "false": ["false", "no"]}
    winner_options = aliases.get(winner.casefold(), [winner])
    loser_options = aliases.get(loser.casefold(), [loser])
    if not any(contains(text, option) for option in winner_options):
        return False
    normalized = text.casefold()
    for option in winner_options:
        normalized = re.sub(_token_pattern(option), "choice_w", normalized)
    for option in loser_options:
        normalized = re.sub(_token_pattern(option), "choice_l", normalized)
    if re.search(r"choice_[wl][^.;!?\n]{0,20}\bnot\s+(?:more|less|the winner)\b", normalized):
        return False
    if _direction(normalized, "choice_l", "choice_w"):
        return False
    # A direct answer naming only the winner is unambiguous unless negated.
    if "choice_l" not in normalized:
        return not re.search(
            r"\b(?:not|isn't|isnt|never)\b[^.;!?\n]{0,35}choice_w"
            r"|choice_w\s+(?:is\s+)?not\b", normalized,
        )
    if not _direction(normalized, "choice_w", "choice_l"):
        return False
    # If counts are reported for each option, they must agree with the table.
    counts = check.get("counts", {})
    for marker, choice in [("choice_w", winner), ("choice_l", loser)]:
        if choice not in counts:
            continue
        for mention in re.finditer(marker, normalized):
            prefix = normalized[:mention.start()]
            preceding = list(_numbers(prefix))
            if preceding:
                number_match, number_value = preceding[-1]
                if (re.fullmatch(r"\s*(?:(?:rows?|entries|times|for|of)\s+)?",
                                 prefix[number_match.end():])
                        and number_value != Decimal(str(counts[choice]))):
                    return False
            tail = re.split(r"choice_[wl]|[;\n]", normalized[mention.end():], maxsplit=1)[0]
            match = re.match(r"\s*(?:\(\s*|:\s*|(?:appears|occurs|has|count(?:\s+is)?)\s+)?", tail)
            remainder = tail[match.end():]
            number = next(_numbers(remainder), None)
            if number and number[0].start() == 0 and number[1] != Decimal(str(counts[choice])):
                return False
    return True


def _preference(text, value):
    """Check the six standing requests, preserving their scope and direction."""
    text = text.casefold().replace("'", "")
    requirements = {
        "log every price change in the notes before updating a table": [
            r"\b(?:price|prices)\b", r"\bchanges?\b", r"\bnotes?\b",
            r"\b(?:before|first|prior)\b", r"\b(?:updat\w*|edit\w*)\b", r"\btables?\b",
            r"\b(?:every|each|all|always)\b",
        ],
        "keep summaries under three sentences": [
            r"\bsummar(?:y|ies)\b", r"\bsentences?\b",
            r"\b(?:under|fewer than|less than|below)\s+(?:three|3)\b"
            r"|\b(?:at most|max|maximum(?: of)?|no more than|not more than|up to)"
            r"\s+(?:two|2)\b",
        ],
        "write dates as year-month-day everywhere": [
            r"\bdates?\b", r"\b(?:year[ /-]month[ /-]day|yyyy[ /-]mm[ /-]dd|iso(?:\s+8601)?)\b",
            r"\b(?:everywhere|all|always|every|consistently)\b",
        ],
        "flag anything overdue in the finish summary": [
            r"\b(?:flag\w*|highlight\w*|mention\w*|include\w*)\b", r"\boverdue\b",
            r"\b(?:finish|final)\b", r"\bsummar(?:y|ies)\b",
        ],
        "never archive a row without a note entry": [
            r"\barchiv\w*\b", r"\brows?\b", r"\bnotes?\b",
            r"\bnever\b[^.;\n]*\bwithout\b"
            r"|\b(?:do not|dont|cannot|must not)\s+archive\b[^.;\n]*\bwithout\b"
            r"|\b(?:before|first|prior|after|unless|must|always|requires?)\b"
            r"|\bonly(?:\s+\w+){0,3}\s+if\b",
        ],
        "start every digest with the row count": [
            r"\bdigests?\b", r"\brows?\b", r"\b(?:count|total|number)\b",
            r"\b(?:start\w*|begin\w*|lead\w*|first|open\w*)\b",
            r"\b(?:every|each|all|always)\b",
        ],
    }
    patterns = requirements.get(value)
    if patterns is None:
        return facts(text, [value]) and not re.search(r"\b(?:not|never|dont|cannot)\b", text)
    if not all(re.search(pattern, text) for pattern in patterns):
        return False
    if value == "never archive a row without a note entry":
        if re.search(r"\b(?:not|dont|shouldnt)\s+never\s+archive\b", text):
            return False
        if re.search(r"\bwithout\b", text):
            return bool(re.search(
                r"\b(?:never|do not|dont|cannot|must not)\s+archive\b[^.;\n]*\bwithout\b",
                text,
            ))
        if re.search(r"\bunless\b", text):
            return bool(re.search(r"\b(?:never|do not|dont)\s+archive\b[^.;\n]*\bunless\b",
                                  text))
        return not re.search(
            r"\b(?:can|may|should|allowed to)\s+archive\b[^.;\n]*\bwithout\b"
            r"|\b(?:dont|not)\b[^.;\n]*\b(?:note|log)\b", text,
        )
    negation_text = re.sub(r"\bnot more than\s+(?:two|2)\b", "at most two", text)
    return not re.search(r"\b(?:not|never|dont|cannot|shouldnt|wont)\b", negation_text)


def _positive(text, pattern):
    for match in re.finditer(pattern, text):
        before = re.split(r"[.;!?\n]", text[:match.start()])[-1][-45:]
        if not re.search(r"\b(?:not|no|none|nothing|never|without|dont|don't"
                         r"|couldnt|couldn't)\b", before):
            return True
    return False


def _outcome(text, check):
    text = text.casefold()
    status = check["status"]
    if status == "denied":
        valid = re.search(
            r"\b(?:denied|declined|rejected|not approved|did not approve|wasnt approved"
            r"|said no|cannot|could not|blocked|refused)\b", text,
        )
        success = _positive(
            text,
            r"\b(?:successfully|completed|fetched|updated|added|created)\b",
        )
        return bool(valid) and not bool(success)
    if status == "empty":
        valid = bool(re.search(
            r"\b(?:no|zero|0)\s+(?:matching\s+|new\s+|relevant\s+)?"
            r"(?:rows?|entries|records?|items?|matches)\b"
            r"|\b(?:nothing|none|empty|no matches|no entries|no results|no hits)\b"
            r"|\b(?:no|zero)\b[^.;!?\n]{0,60}\bfound\b", text,
        ))
        positive = False
        for match, number in _numbers(text):
            tail = text[match.end():]
            before = re.split(r"[.;!?\n]", text[:match.start()])[-1]
            reported_match = re.match(r"\s+(?:matching\s+(?:rows?|entries)|matches)\b", tail)
            reported_find = (re.search(r"\b(?:found|identified|located)\s*$", before)
                             and re.match(r"\s+(?:rows?|entries)\b", tail))
            if (number > 0 and (reported_match or reported_find)
                    and _positive(text, re.escape(match.group()))):
                positive = True
                break
        return valid and not positive
    if status == "absent":
        return bool(re.search(
            r"\b(?:absent|missing|not found|not present|not listed|no longer present"
            r"|could not find|couldnt find|cannot find|cant find|can't find"
            r"|does not exist|doesnt exist|not in|not yet in|isnt in|isn't in"
            r"|not (?:yet )?(?:been )?(?:added|logged|recorded|entered)"
            r"|hasnt been added|hasn't been added|wasnt added|wasn't added"
            r"|no sign of|no trace of)\b", text,
        )) and not _positive(
            text, r"\b(?:found|present|listed|added|logged)\s+"
            + _token_pattern(check["subject"]),
        )
    raise ValueError(f"Unknown training outcome: {status}")


def _answer(text, check):
    if not _valid_statement(text.casefold()):
        return False
    value = check["value"]
    if isinstance(value, bool) or str(value).casefold() in {"true", "false"}:
        options = ["true", "yes"] if str(value).casefold() == "true" else ["false", "no"]
    else:
        options = [str(value)]
    try:
        numeric = Decimal(str(value))
    except InvalidOperation:
        numeric = None
    if numeric is not None:
        options += [word for word, number in _NUMBERS.items() if number == numeric]
        seen = any(number == numeric for _, number in _numbers(text))
    else:
        seen = any(contains(text, option) for option in options)
    if not seen:
        return False
    if isinstance(value, bool) or str(value).casefold() in {"true", "false"}:
        opposite = "false" if str(value).casefold() == "true" else "true"
        if _positive(text.casefold(), _token_pattern(opposite)):
            return False
    for option in options:
        pattern = _token_pattern(option)
        if re.search(r"\b(?:not|isnt|isn't|never)\s+" + pattern, text.casefold()):
            return False
    # An explicit column/value answer provides a useful contradiction signal.
    column = check.get("column")
    if column and numeric is not None:
        match = re.search(_token_pattern(column) + r"\s*(?:is|=|:|was)?\s*", text.casefold())
        if match:
            number = next(_numbers(text[match.end():]), None)
            if number and number[0].start() == 0 and number[1] != numeric:
                return False
    elif column:
        match = re.search(
            _token_pattern(column) + r"\s*(?:is|=|:|was|reads|shows|says)\s*", text.casefold(),
        )
        if match:
            clause = re.split(r"[,.;!?\n]", text[match.end():], maxsplit=1)[0]
            if not any(contains(clause, option) for option in options):
                return False
    if numeric is None:
        mentions = [match for option in options
                    for match in re.finditer(_token_pattern(option), text.casefold())]
        if mentions and all(re.search(
            r"\b(?:previously|previous|prior|old|earlier|historical|yesterday|last)\b",
            re.split(r"[,.;!?\n]", text[:match.start()])[-1], re.IGNORECASE,
        ) for match in mentions):
            return False
    return True


def _action(text):
    return not re.search(
        r"\b(?:cancel\w*|ignore|disregard|do not|dont|don't|no need to)\b", text.casefold(),
    )


_ATOM_RE = re.compile(
    r"(?<!\w)(?:\d{1,2}:\d{2}(?:\s*[ap]m)?|\d{1,2}\s*[ap]m"
    r"|[+-]?\d+(?:\.\d+)?|twenty[ -](?:one|two|three|four)|"
    + "|".join(reversed(_WORDS)) + r")(?!\w|\.\d|-[a-z])", re.IGNORECASE,
)


def _atoms(text):
    atoms = []
    for match in _ATOM_RE.finditer(text):
        token = match.group().casefold()
        word = token.replace("-", " ")
        if re.search(r"[ap]m|:", token):
            value = re.sub(r"\s+", "", token)
        else:
            value = _NUMBERS[word] if word in _NUMBERS else Decimal(token)
        atoms.append(value)
    return atoms


def _segments(text):
    return [part.strip() for part in re.split(
        r"\n|;|(?<=[.!?])\s+|\s+and\s+(?=[A-Z])|,\s*(?=[A-Z])", text,
    ) if part.strip()]


def _negated(text, token):
    return bool(re.search(
        r"\b(?:not|never|no|isnt|isn't|wasnt|wasn't|dont|don't|didnt|didn't)\s+"
        r"(?:\w+\s+){0,2}" + _token_pattern(token), text.casefold(),
    ))


def _relations(text, sources, anchors=None):
    """Preserve local number/time associations and explicit fact polarity.

    This is a bounded guard for source statements, not a general entailment
    engine. Clause order may change; counts/range endpoints stay associated with
    the material words in their source clause.
    """
    candidates = _segments(text)
    anchor_tokens = None if anchors is None else {
        token for anchor in anchors
        for token in re.findall(r"\w+(?:[-.]\w+)*", str(anchor).casefold())
    }
    for source in sources:
        for segment in _segments(str(source)):
            tokens = re.findall(r"\w+(?:[-.]\w+)*", segment.casefold())
            material = [token for token in tokens if token not in _STOP_WORDS
                        and token not in {"not", "never", "no", "until", "through", "then"}]
            numeric = _atoms(segment)
            context = [token for token in material if not _ATOM_RE.fullmatch(token)]
            if anchor_tokens is not None:
                context = [token for token in context if token in anchor_tokens]
            relevant = [candidate for candidate in candidates if facts(candidate, context)]
            if numeric:
                matching = False
                for candidate in relevant:
                    observed = iter(_atoms(candidate))
                    if all(any(actual == expected for actual in observed) for expected in numeric):
                        matching = True
                        break
                if not matching:
                    return False
            for token in material:
                if _ATOM_RE.fullmatch(token):
                    continue
                negative = _negated(segment, token)
                local = relevant if relevant else candidates
                if negative:
                    if not any(_negated(candidate, token) for candidate in local):
                        return False
                elif any(_negated(candidate, token) for candidate in local):
                    return False
    return True


def failures(scenario, messages):
    """Return failed scenario checks without changing runtime assertions."""
    checks = scenario.get("checks", [])
    if not checks:
        return []
    calls = _calls(messages)
    failed = []
    for check in checks:
        kind = check["kind"]
        passed = False
        if kind == "count":
            texts = _texts(calls, check)
            if check.get("tool") == "note_write":
                texts = texts[-1:]
            passed = any(_count(text, check) for text in texts)
        elif kind == "compare":
            passed = any(_comparison(text, check) for text in _texts(calls, check))
        elif kind == "list":
            passed = any(all(contains(text, value) for value in check["wanted"])
                         and not any(contains(text, value) for value in check.get("excluded", []))
                         for text in _texts(calls, check))
        elif kind == "text":
            texts = (_note_texts(scenario, calls, check) if check.get("tool") == "note_write"
                     else _texts(calls, check))
            excluded = check.get("excluded", [])
            passed = any(facts(text, check["facts"])
                         and _relations(text, check.get("sources", []), check["facts"])
                         and not any(contains(text, item) for item in excluded)
                         for text in texts)
        elif kind == "task":
            passed = any(call.get("tool") == "task_add"
                         and call["args"].get("to") == check["to"]
                         and facts(call["args"].get("title", "") + "\n"
                                   + call["args"].get("details", ""), check["facts"])
                         and _action(call["args"].get("title", "") + "\n"
                                     + call["args"].get("details", ""))
                         and _relations(call["args"].get("title", "") + "\n"
                                        + call["args"].get("details", ""),
                                        check.get("sources", []), check["facts"])
                         for call in calls)
        elif kind == "preference":
            texts = _texts(calls, {"tool": "remember"})
            passed = bool(texts) and all(_preference(text, check["value"]) for text in texts)
        elif kind == "follow_up":
            passed = any(call.get("tool") == "follow_up"
                         and contains(call["args"].get("note", ""), check["table"])
                         and (not check.get("marker")
                              or contains(call["args"].get("note", ""), check["marker"]))
                         for call in calls)
        elif kind == "state":
            passed = _state(scenario, calls, check)
        elif kind == "outcome":
            passed = any(_outcome(text, check) for text in _texts(calls, check))
        elif kind == "answer":
            passed = any(_answer(text, check) for text in _texts(calls, check))
        elif kind == "format":
            for text in _note_texts(scenario, calls, check):
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                if check["style"] == "one_line":
                    passed |= len(lines) == 1
                elif check["style"] == "bullets":
                    items = [line for line in lines if not re.match(r"^#{1,6}\s+", line)]
                    passed |= bool(items) and all(re.match(r"^(?:[-*]|\d+[.)])\s+", line)
                                                 for line in items)
                else:
                    raise ValueError(f"Unknown training note format: {check['style']}")
        elif kind == "notes":
            allowed = set(check["allowed"])
            passed = all(call.get("tool") != "note_write"
                         or call["args"].get("title") in allowed for call in calls)
        elif kind == "order":
            tools = [call.get("tool") for call in calls]
            before = next((index for index, tool in enumerate(tools)
                           if tool == check["before"]), None)
            passed = before is not None and all(index > before for index, tool in enumerate(tools)
                                               if tool in check["after"])
        else:
            raise ValueError(f"Unknown training check kind: {kind}")
        if not passed:
            failed.append(check)
    return failed
