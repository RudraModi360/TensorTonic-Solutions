import heapq
def schedule_pipeline(tasks, resource_budget):
    """
    Schedule ETL tasks respecting dependencies and resource limits.
    """
    task_map = {t["name"]: t for t in tasks}

    # Track states
    completed = set()
    started = set()
    
    # Min-heap for running tasks: (end_time, task_name)
    running = []
    
    # Result: (task_name, start_time)
    schedule = []

    current_time = 0
    current_resources = 0

    while len(completed) < len(tasks):
        # 1. Complete finished tasks
        while running and running[0][0] <= current_time:
            end_time, name = heapq.heappop(running)
            completed.add(name)
            current_resources -= task_map[name]["resources"]

        # 2. Find ready tasks
        ready = []
        for t in tasks:
            name = t["name"]
            if name in started:
                continue
            if all(dep in completed for dep in t["depends_on"]):
                ready.append(name)

        ready.sort()  # alphabetical order

        # 3. Try scheduling tasks greedily
        scheduled_any = False
        for name in ready:
            t = task_map[name]
            if current_resources + t["resources"] <= resource_budget:
                # Start task
                started.add(name)
                schedule.append((name, current_time))
                end_time = current_time + t["duration"]
                heapq.heappush(running, (end_time, name))
                current_resources += t["resources"]
                scheduled_any = True

        # 4. Advance time
        if running:
            # Jump to next completion if nothing new scheduled
            if not scheduled_any:
                current_time = running[0][0]
        else:
            break  # safety (should not happen in valid DAG)

    # Sort final output
    return sorted(schedule, key=lambda x: (x[1], x[0]))