from __future__ import annotations

NAVIGATION_TEMPLATES = [
    "Please find {object}.",
    "Kindly locate {object}.",
    "Could you lead me to {object}?",
    "Your task is to get close to {object}.",
]

MANIPULATION_INSIDE_TEMPLATES = [
    "Please find {object} and put it into {target}.",
    "Kindly locate {object} and place it in {target}.",
    "Could you retrieve {object} and place it into {target}?",
    "Your task is to get {object} and move it to {target}.",
]

MANIPULATION_ON_TEMPLATES = [
    "Please find {object} and put it on {target}.",
    "Kindly locate {object} and place it on {target}.",
    "I need you to search for {object} and position it on {target}.",
    "Your task is to grab {object} and move it onto {target}.",
]

MULTI_INSIDE_TEMPLATES = [
    "Kindly locate {object1} and {object2}, and place them inside {target}.",
    "I need you to search for {object1} and {object2} and then position them into {target}.",
    "Your task is to gather {object1} and {object2} and subsequently put them into {target}.",
    "Please find {object1} and {object2}, then put them into {target}.",
]

MULTI_ON_TEMPLATES = [
    "Kindly locate {object1} and {object2}, and place them on {target}.",
    "I need you to search for {object1} and {object2} and then position them onto {target}.",
    "Your task is to gather {object1} and {object2} and subsequently put them onto {target}.",
    "Please find {object1} and {object2}, then put them on {target}.",
]

LONG_HORIZON_TEMPLATES = [
    "Organize the scene in order: {placements}.",
    "Resolve the access constraints, then complete these placements in order: {placements}.",
    "Carry out the ordered cleanup plan: {placements}.",
]


def choose_template(rng, templates: list[str]) -> str:
    return templates[rng.randrange(len(templates))]
