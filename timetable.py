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
# # scoruri pt profi si materii

def parse_interval(interval : str):
    '''
    Se parsează un interval de forma "Ora1 - Ora2" în cele 2 componente.
    '''

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
        return len(self.subjects_remaining) == 0 and self.total_violations == 0

    def generate_neighbors(self):
        # Generate neighboring states by assigning subjects to empty slots
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
                                        new_timetable = copy.deepcopy(self.timetable)
                                        new_timetable[day][interval][classroom] = (professor, subject)
                                        new_profs_in_interval = copy.deepcopy(self.profs_in_interval)
                                        new_profs_in_interval[day][interval].append(professor)
                                        new_subjects_remaining = copy.deepcopy(self.subjects_remaining)

                                        new_subjects_remaining[subject] -= timetable_specs[SALI][classroom][CAPACITATE]
                                        if new_subjects_remaining[subject] <= 0:
                                            del new_subjects_remaining[subject]
                                        new_prof_hrs = copy.deepcopy(self.prof_hrs)
                                        new_prof_hrs[professor] += 1

                                        new_profs_cost = copy.deepcopy(self.prof_cost)
                                        if len(timetable_specs[PROFESORI][professor][MATERII]) > 1:
                                            for materie in timetable_specs[PROFESORI][professor][MATERII]:
                                                if materie in new_subjects_remaining:
                                                    new_profs_cost += 2
                                        
                                        new_classroom_cost = copy.deepcopy(self.classroom_cost)
                                        if len(timetable_specs[SALI][classroom][MATERII]) > 1:
                                            for materie in timetable_specs[SALI][classroom][MATERII]:
                                                if materie in new_subjects_remaining:
                                                    new_classroom_cost += 2

                                        new_cost = self.cost + 1
                                        new_state = State(new_timetable, timetable_specs, new_subjects_remaining, new_prof_hrs, new_profs_in_interval, new_cost, new_profs_cost, new_classroom_cost)
                                        new_state.total_violations = new_state.check_optional_constraints(day, interval, subject, professor, classroom)
                                        neighbors.append(new_state)

                                            
        return neighbors

    def is_valid_assignment(self, day, interval, subject, professor, classroom):
        # Check constraints to ensure this assignment is valid
        if self.timetable[day][interval][classroom] is not None:
            return False  # Room already occupied in this interval
        if professor not in timetable_specs[PROFESORI]:
            return False
        if subject not in timetable_specs[MATERII]:
            return False
        if subject not in timetable_specs[PROFESORI][professor][MATERII]:
            return False  # Professor doesn't teach this subject
        if subject not in timetable_specs[SALI][classroom][MATERII]:
            return False  # Room doesn't support this subject
        if self.prof_hrs[professor] + 1 > 7:
            return False
        return True
    
    def check_optional_constraints(self, day, interval, subject, professor, classroom):
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
        return self.total_violations == other.total_violations
    
    def __hash__(self):
        return hash(str(self.timetable))

class AStarSearch:
    def __init__(self, initial_state):
        self.initial_state = initial_state

    def heuristic(self, state):
        heuristic_value = 0
        for subject in state.subjects_remaining:
            heuristic_value += state.subjects_remaining[subject]

        heuristic_value += 2 * state.prof_cost
        heuristic_value += 2 * state.classroom_cost
        heuristic_value += 10 * state.total_violations
        
        return heuristic_value

    def search(self):
        open_set = []
        heapq.heappush(open_set, (self.heuristic(self.initial_state), self.initial_state))

        closed_set = set()
        while open_set:
            h, current_state = heapq.heappop(open_set)
            print(current_state.timetable)
            print(h)
            print(current_state.subjects_remaining)

            if current_state.is_goal():
                return current_state

            closed_set.add(current_state)

            # Generate neighbors and evaluate them
            for neighbor in current_state.generate_neighbors():
                if neighbor not in closed_set:
                    heapq.heappush(open_set, (self.heuristic(neighbor), neighbor))
            

        return None  # No valid solution found


# Entry point for the A* algorithm
def get_timetable(timetable_specs):
    
    # Initialize an empty timetable structure
    timetable = {day: {eval(interval): {classroom: None for classroom in timetable_specs[SALI]} for interval in timetable_specs[INTERVALE]} for day in timetable_specs[ZILE]}
    
    # Initialize the subjects remaining to be assigned
    subjects_remaining = {subject: timetable_specs[MATERII][subject] 
                          for subject in timetable_specs[MATERII]}
    
    prof_hrs = {prof: 0 for prof in timetable_specs[PROFESORI]}

    profs_in_interval = {day: {eval(interval): [] for interval in timetable_specs[INTERVALE]} for day in timetable_specs[ZILE]}
    
    # Create an initial state with the empty timetable
    initial_state = State(timetable, timetable_specs, subjects_remaining, prof_hrs, profs_in_interval, 0, 0, 0)
    
    # Create and execute the AStarSearch
    astar_search = AStarSearch(initial_state)
    result = astar_search.search()
    
    if result:
        # Format the output timetable as needed
        return result.timetable
    else:
        raise Exception("No valid timetable found")

# Implement the Hill-Climbing algorithm for timetable scheduling
class HillClimbing:
    def __init__(self, initial_state):
        self.initial_state = initial_state

    def search(self):
        current_state = self.initial_state
        current_cost = current_state.evaluate()

        while True:
            # Generate all neighbors
            neighbors = current_state.generate_neighbors()
            if not neighbors:
                break  # No neighbors, stop searching

            # Find the best neighbor (with the lowest cost)
            best_neighbor = min(neighbors, key=lambda s: s.evaluate())
            best_cost = best_neighbor.evaluate()

            if best_cost < current_cost:
                # If a better neighbor is found, move to it
                current_state = best_neighbor
                current_cost = best_cost
            else:
                break  # No improvement, stop searching

        return current_state  # Return the best state found

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

    # timetable = {day : {eval(interval) : {} for interval in timetable_specs[INTERVALE]} for day in timetable_specs[ZILE]}

    # initial_state = State(timetable, 0, timetable_specs[PROFESORI])

    if sys.argv[2] == 'astar':
        timetable = get_timetable(timetable_specs)
    # elif sys.argv[2] == 'hc':
    #     timetable = HillClimbing(initial_state).search()

    if debug_flag:
        print(pretty_print_timetable(timetable, input_name))

    print('\n----------- Constrângeri obligatorii -----------')
    constrangeri_incalcate = check_mandatory_constraints(timetable, timetable_specs)

    print(f'\n=>\nS-au încălcat {constrangeri_incalcate} constrângeri obligatorii!')

    print('\n----------- Constrângeri optionale -----------')
    constrangeri_optionale = check_optional_constraints(timetable, timetable_specs)
    
    print(f'\n=>\nS-au încălcat {constrangeri_optionale} constrângeri optionale!\n')

