from collections import defaultdict, deque

def min_semesters(num_courses: int, prerequisites: list[list[int]]) -> int:
    """
    This function calculates the minimum number of semesters needed to complete all courses
    considering the given prerequisites. It uses a topological sorting approach with a
    breadth-first search (BFS) to handle the semester constraint. The time complexity is O(N + E),
    where N is the number of courses and E is the number of prerequisites.

    Args:
    num_courses (int): The total number of courses.
    prerequisites (list[list[int]]): A list of pairs of courses where the first course is a prerequisite of the second course.

    Returns:
    int: The minimum number of semesters needed to complete all courses. Returns -1 if it's impossible to finish all courses due to cyclic prerequisites.
    """
    
    # Create a graph where each course is a node, and the prerequisites are directed edges between the nodes.
    graph = defaultdict(list)
    in_degree = {i: 0 for i in range(num_courses)}
    
    # Initialize the in-degree of each node based on the prerequisites.
    for course, prerequisite in prerequisites:
        graph[prerequisite].append(course)
        in_degree[course] += 1
    
    # Initialize a queue with nodes that have an in-degree of 0 (no prerequisites).
    queue = deque([course for course in in_degree if in_degree[course] == 0])
    
    # Initialize the number of semesters and the number of processed courses.
    semesters = 0
    processed_courses = 0
    
    # Perform BFS.
    while queue:
        # Process all courses in the current level.
        for _ in range(len(queue)):
            course = queue.popleft()
            processed_courses += 1
            
            # Decrease the in-degree of the neighbors of the current course.
            for neighbor in graph[course]:
                in_degree[neighbor] -= 1
                
                # If a neighbor's in-degree becomes 0, add it to the queue.
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Increment the semester counter if we have processed courses in the current level.
        if processed_courses > 0:
            semesters += 1
    
    # If we cannot process all nodes, it means there's a cycle, and we return -1.
    if processed_courses != num_courses:
        return -1
    
    # If there are no prerequisites, all courses can be taken in a single semester.
    if semesters == 0 and num_courses > 0:
        return 1
    
    return semesters
