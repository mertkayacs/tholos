from fnmatch import fnmatchcase
from urllib.parse import urlsplit

from tholos import workspace as w


def target(tool: str, args: dict) -> str:
    if tool.startswith("table_"):
        return args["table"]
    if tool.startswith("note_"):
        return args["title"]
    if tool == "task_add":
        return args["to"]
    if tool == "web_fetch":
        return (urlsplit(args["url"]).hostname or "").rstrip(".").lower()
    return "*"


def check(db: w.DB, agent_name: str, tool: str, target: str) -> str:
    matches = [
        rule["decision"]
        for rule in w.list_rules(db)
        if rule["agent"].casefold() in {"*", agent_name.casefold()}
        and rule["tool"] == tool
        and fnmatchcase(target.casefold(), rule["match"].casefold())
    ]
    for decision in ("deny", "ask", "allow"):
        if decision in matches:
            return decision
    return "ask" if tool == "web_fetch" else "allow"
