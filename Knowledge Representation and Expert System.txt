"""
Medical diagnosis expert system (propositional Horn clauses).
Rule format: (set of premises, conclusion)
"""
RULES = [
    ({"fever", "cough"}, "flu_suspect"),
    ({"flu_suspect", "body_ache"}, "flu"),
    ({"sneezing", "runny_nose"}, "cold"),
    ({"flu"}, "rest_and_fluids"),
    ({"flu", "high_risk"}, "see_doctor"),
    ({"cold"}, "take_antihistamine"),
]


def forward_chaining(facts, goal=None):
    """Data-driven: fire rules until no new facts are derived."""
    facts = set(facts)
    changed = True
    while changed:
        changed = False
        for premises, conclusion in RULES:
            if premises <= facts and conclusion not in facts:
                facts.add(conclusion)
                print(f"  Fired: {sorted(premises)} -> {conclusion}")
                changed = True
                if conclusion == goal:
                    return True, facts
    return (goal in facts if goal else True), facts


def backward_chaining(goal, facts, depth=0):
    """Goal-driven: prove goal from facts or by proving rule premises."""
    pad = "  " * (depth + 1)
    if goal in facts:
        print(f"{pad}{goal}: known fact")
        return True
    for premises, conclusion in RULES:
        if conclusion == goal:
            print(f"{pad}{goal}: try rule {sorted(premises)}")
            if all(backward_chaining(p, facts, depth + 1) for p in premises):
                return True
    print(f"{pad}{goal}: cannot be proved")
    return False


facts = {"fever", "cough", "body_ache", "high_risk"}
print("Facts:", sorted(facts))

print("\nForward Chaining:")
_, derived = forward_chaining(facts)
print("  Derived:", sorted(derived - facts))

print("\nBackward Chaining (goal: see_doctor):")
print("  Result:", backward_chaining("see_doctor", facts))

print("\nBackward Chaining (goal: take_antihistamine):")
print("  Result:", backward_chaining("take_antihistamine", facts))