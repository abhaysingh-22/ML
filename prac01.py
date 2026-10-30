def smart_home_agent(temp, is_dark, person_in_room):
    actions = []

    # Rules for temperature
    if temp > 30:
        actions.append("Turn ON AC")
    elif temp < 18:
        actions.append("Turn ON Heater")
    else:
        actions.append("Temperature OK")

    # Rules for light
    if person_in_room and is_dark:
        actions.append("Turn ON Light")
    elif not person_in_room:
        actions.append("Turn OFF Light")

    return actions




percepts = [
    (35, True,  True),
    (15, True,  True),
    (24, False, True),
    (28, True,  False),
    (32, False, False),
]

for p in percepts:
    result = smart_home_agent(p[0], p[1], p[2])
    print("Percept:", p, "->", result)
    
    
    
    
expected_first_action = ["Turn ON AC", "Turn ON Heater", "Temperature OK",
                         "Temperature OK", "Turn ON AC"]

score = 0
for i in range(len(percepts)):
    p = percepts[i]
    action = smart_home_agent(p[0], p[1], p[2])[0]
    if action == expected_first_action[i]:
        score = score + 1

print("Score:", score, "out of", len(percepts))