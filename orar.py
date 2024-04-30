import random
import yaml
import sys
from utils import read_yaml_file, acces_yaml_attributes, get_profs_initials, allign_string_with_spaces, pretty_print_timetable_aux_zile, pretty_print_timetable_aux_intervale, pretty_print_timetable
from check_constraints import check_mandatory_constraints, check_optional_constraints
import heapq
import copy

INTERVALE = 'Intervale'
ZILE = 'Zile'
MATERII = 'Materii'
PROFESORI = 'Profesori'
SALI = 'Sali'
CAPACITATE = 'Capacitate'
CONSTRANGERI = 'Constrangeri'

total_states_generated = 0
total_states_explored = 0

def parse_interval(interval : str):
    intervals = interval.split('-')
    return int(intervals[0].strip()), int(intervals[1].strip())

# Class to represent the state of the timetable
class State:
    def __init__(self, timetable, timetable_specs, subjects_remaining, prof_hrs, profs_in_interval, cost, prof_cost, classroom_cost):
        self.timetable = timetable
        self.timetable_specs = timetable_specs
        self.cost = cost
        self.total_violations = 0
        self.prof_cost = prof_cost
        self.classroom_cost = classroom_cost
        self.prof_hrs = prof_hrs
        self.subjects_remaining = subjects_remaining
        self.profs_in_interval = profs_in_interval

    def get_cost(self):
        return self.cost

    def is_goal(self):
        return len(self.subjects_remaining) == 0

    def generate_neighbors(self):
        # Generate all possible neighbors of the current state
        neighbors = []
        timetable_specs = self.timetable_specs
        timetable = self.timetable
        subjects_remaining = self.subjects_remaining
        for day in timetable_specs[ZILE]:
            for interval in timetable[day]:
                for subject in subjects_remaining:
                    if subjects_remaining[subject] > 0:
                        for classroom in timetable_specs[SALI]:
                            for professor in timetable_specs[PROFESORI]:
                                if professor not in self.profs_in_interval[day][interval] and timetable[day][interval][classroom] is None:
                                    if self.is_valid_assignment(day, interval, subject, professor, classroom):
                                        # Copy the current state and update it with the new assignment
                                        new_timetable = copy.deepcopy(self.timetable)
                                        new_timetable[day][interval][classroom] = (professor, subject)
                                        new_profs_in_interval = copy.deepcopy(self.profs_in_interval)
                                        new_profs_in_interval[day][interval].append(professor)
                                        new_subjects_remaining = copy.deepcopy(self.subjects_remaining)

                                        # Update the remaining subjects to be scheduled
                                        new_subjects_remaining[subject] -= timetable_specs[SALI][classroom][CAPACITATE]
                                        if new_subjects_remaining[subject] <= 0:
                                            del new_subjects_remaining[subject]
                                        new_prof_hrs = copy.deepcopy(self.prof_hrs)
                                        new_prof_hrs[professor] += 1

                                        # Update the prof and classroom costs
                                        new_profs_cost = copy.deepcopy(self.prof_cost)
                                        if len(timetable_specs[PROFESORI][professor][MATERII]) > 1:
                                            for materie in timetable_specs[PROFESORI][professor][MATERII]:
                                                if materie in new_subjects_remaining:
                                                    new_profs_cost += 1
                                        
                                        new_classroom_cost = copy.deepcopy(self.classroom_cost)
                                        if len(timetable_specs[SALI][classroom][MATERII]) > 1:
                                            for materie in timetable_specs[SALI][classroom][MATERII]:
                                                if materie in new_subjects_remaining:
                                                    new_classroom_cost += 1

                                        # Create the new state and add it to the neighbors
                                        new_cost = self.cost + 1
                                        new_state = State(new_timetable, timetable_specs, new_subjects_remaining, new_prof_hrs, new_profs_in_interval, new_cost, new_profs_cost, new_classroom_cost)
                                        new_state.total_violations = new_state.check_optional_constraints(day, interval, subject, professor, classroom)
                                        neighbors.append(new_state)
                                        global total_states_generated
                                        total_states_generated += 1

                                            
        return neighbors

    def is_valid_assignment(self, day, interval, subject, professor, classroom):
        # Check constraints to ensure this assignment is valid
        if self.timetable[day][interval][classroom] is not None:
            return False  # Room already occupied in this interval
        if professor not in timetable_specs[PROFESORI]:
            return False # Professor not in the timetable
        if subject not in timetable_specs[MATERII]:
            return False # Subject not in the timetable
        if subject not in timetable_specs[PROFESORI][professor][MATERII]:
            return False  # Professor doesn't teach this subject
        if subject not in timetable_specs[SALI][classroom][MATERII]:
            return False  # Room doesn't support this subject
        if self.prof_hrs[professor] + 1 > 7:
            return False # Professor already has 7 hours scheduled
        return True
    
    def check_optional_constraints(self, day, interval, subject, professor, classroom):
        # Check optional constraints and return the number of violations
        violations = 0
        for prof in self.timetable_specs[PROFESORI]:
            for const in self.timetable_specs[PROFESORI][prof][CONSTRANGERI]:
                if const[0] != '!':
                    continue
                else:
                    const = const[1:]

                    if const in self.timetable_specs[ZILE]:
                        day = const
                        if day in self.timetable:
                            for interval in self.timetable[day]:
                                for room in self.timetable[day][interval]:
                                    if self.timetable[day][interval][room]:
                                        crt_prof, _ = self.timetable[day][interval][room]
                                        if prof == crt_prof:
                                            violations += 1
                
                    elif '-' in const:
                        interval = parse_interval(const)
                        start, end = interval

                        if start != end - 2:
                            intervals = [(i, i + 2) for i in range(start, end, 2)]
                        else:
                            intervals = [(start, end)]


                        for day in self.timetable:
                            for interval in intervals:
                                if interval in self.timetable[day]:
                                    for room in self.timetable[day][interval]:
                                        if self.timetable[day][interval][room]:
                                            crt_prof, _ = self.timetable[day][interval][room]
                                            if prof == crt_prof:
                                                violations += 1
        return violations
    
    def __lt__(self, other):
        return self.total_violations < other.total_violations
    
    def __eq__(self, other):
        return str(self.timetable) == str(other.timetable)
    
    def __hash__(self):
        return hash(str(self.timetable))

class AStar:
    def __init__(self, initial_state):
        self.initial_state = initial_state

    def heuristic(self, state):
        # Heuristic function to estimate the cost to reach the goal state
        heuristic_value = 0
        for subject in state.subjects_remaining:
            heuristic_value += state.subjects_remaining[subject]

        heuristic_value += 4 * state.prof_cost
        heuristic_value += 4 * state.classroom_cost
        heuristic_value += 10 * state.total_violations
        
        return heuristic_value

    def search(self):
        open_set = []
        heapq.heappush(open_set, (self.heuristic(self.initial_state), self.initial_state))

        closed_set = set()
        while open_set:
            h, current_state = heapq.heappop(open_set)

            if current_state in closed_set:
                continue

            global total_states_explored
            total_states_explored += 1

            if current_state.is_goal():
                return current_state

            closed_set.add(current_state)

            # Generate neighbors and evaluate them
            for neighbor in current_state.generate_neighbors():
                if neighbor not in closed_set:
                    heapq.heappush(open_set, (neighbor.cost + self.heuristic(neighbor), neighbor))
            

        return None  # No valid solution found


def get_timetable(timetable_specs, algorithm):
    
    # Initialize an empty timetable
    timetable = {day: {eval(interval): {classroom: None for classroom in timetable_specs[SALI]} for interval in timetable_specs[INTERVALE]} for day in timetable_specs[ZILE]}
    
    # Initialize the subjects remaining to be assigned
    subjects_remaining = {subject: timetable_specs[MATERII][subject] 
                          for subject in timetable_specs[MATERII]}
    
    # Initialize the hours for each professor
    prof_hrs = {prof: 0 for prof in timetable_specs[PROFESORI]}

    # Initialize the professors in each interval
    profs_in_interval = {day: {eval(interval): [] for interval in timetable_specs[INTERVALE]} for day in timetable_specs[ZILE]}
    
    # Create an initial state with the empty timetable
    initial_state = State(timetable, timetable_specs, subjects_remaining, prof_hrs, profs_in_interval, 0, 0, 0)
    
    if algorithm == 'astar':
        astar_search = AStar(initial_state)
        result = astar_search.search()
    else:
        result = HillClimbing.random_restart_hill_climbing(initial_state)
    
    if result:
        return result.timetable
    else:
        raise Exception("No valid timetable found")

class HillClimbing:
    def __init__(self, initial_state):
        self.initial_state = initial_state

    def total_cost(self):
        # Calculate the total cost of the current state
        state = self.initial_state
        total_cost_value = 0
        if len(state.subjects_remaining) != 0:
            for subject in state.subjects_remaining:
                total_cost_value += state.subjects_remaining[subject]

        total_cost_value += 4 * state.prof_cost
        total_cost_value += 4 * state.classroom_cost
        total_cost_value += 45 * state.total_violations
        
        return total_cost_value
    
    def search(self, max_iters=100):
        current_state = self.initial_state
        current_cost = self.total_cost()

        global total_states_explored
        total_states_explored += 1
        global total_states_generated

        for _ in range(max_iters):
            best_neighbor = None
            searching = True

            # Find the best neighbor (with the lowest cost)
            for neighbor in current_state.generate_neighbors():
                total_states_generated += 1
                if HillClimbing(neighbor).total_cost() <= current_cost:
                    best_neighbor = neighbor
                    searching = False
            
            if best_neighbor is not None:
                best_cost = HillClimbing(best_neighbor).total_cost()

            if not searching:
                # If a better neighbor is found, move to it
                current_state = best_neighbor
                current_cost = best_cost
                total_states_explored += 1
            else:
                break  # No improvement, stop searching

        return current_state
    
    @staticmethod
    def random_restart_hill_climbing(initial, max_restarts=10, run_max_iters=100):
        total_iters, total_states = 0, 0
        restarts = 0

        # Initial state as the starting point
        current_state = initial

        neighbors = current_state.generate_neighbors()

        while restarts < max_restarts:
            final_state = HillClimbing(current_state).search(run_max_iters)
            
            total_iters += run_max_iters
            total_states += 1
            
            if final_state.is_goal():
                return final_state
            
            restarts += 1

            # Generate a new random state based on the neighbors of the initial state
            current_state = random.choice(neighbors)

        return None


if __name__ == '__main__':

    if len(sys.argv) == 1 or len(sys.argv) == 2:
        print('\nSe rulează de exemplu:\n\npython3 timetable.py orar_mic_exact astar/hc\n')
        sys.exit(0)

    if sys.argv[1] == '-h':
        print('\nSe rulează de exemplu:\n\npython3 timetable.py orar_mic_exact astar/hc\n')

    filename = sys.argv[1]

    input_name = f'inputs/{filename}.yaml'
    output_name = f'outputs/{filename}_timetable.txt'

    timetable_specs = read_yaml_file(input_name)

    debug_flag = True

    if sys.argv[2] == 'astar':
        timetable = get_timetable(timetable_specs, 'astar')
    elif sys.argv[2] == 'hc':
        timetable = get_timetable(timetable_specs, 'hc')

    if debug_flag:
        print(pretty_print_timetable(timetable, input_name))

    print('\n----------- Constrângeri obligatorii -----------')
    constrangeri_incalcate = check_mandatory_constraints(timetable, timetable_specs)

    print(f'\n=>\nS-au încălcat {constrangeri_incalcate} constrângeri obligatorii!')

    print('\n----------- Constrângeri optionale -----------')
    constrangeri_optionale = check_optional_constraints(timetable, timetable_specs)
    
    print(f'\n=>\nS-au încălcat {constrangeri_optionale} constrângeri optionale!\n')

    print(f'\n=>\nS-au generat un total de {total_states_generated} stări!')
    print(f'\n=>\nS-au explorat un total de {total_states_explored} stări!')

