#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
From data (input-output pairings),
and a set of operators and operands as the only starting point,
write a program that will evolve programmatic solutions,
which take in inputs and generate outputs.

Each program will have 1 numeric input and 1 numeric output.
This is much like regression in our simple case,
though can be generalized much further,
toward arbitrarily large and complex programs.

This assignment is mostly open-ended,
with a couple restrictions:

# DO NOT MODIFY >>>>
Do not edit the sections between these marks below.
# <<<< DO NOT MODIFY
"""

# %%
import random
from typing import Optional
from typing import TypedDict
import math
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt


# import json

# import math
# import datetime
# import subprocess


# DO NOT MODIFY >>>>
# First, what should our representation look like?
# Is there any modularity in adjacency?
# What mechanisms capitalize on such modular patterns?
OPERATORS = "+-/*"


class Node:
    """
    Example prefix formula:
    Y = ( * ( + 20 45 ) ( - 56 X ) )
    This is it's tree:
       *
      /  \
    +     -
    / \   / \
    20 45 56  X

    root = Node(
        data="*",
        left=Node(data="+", left=Node("20"), right=Node("45")),
        right=Node(data="-", left=Node("56"), right=Node("X")),
    )
    """

    def __init__(
        self, data: str, left: Optional["Node"] = None, right: Optional["Node"] = None
    ) -> None:
        self.data = data
        self.left = left
        self.right = right


class Individual(TypedDict):
    """Type of each individual to evolve"""

    genome: Node
    fitness: float


Population = list[Individual]


class IOpair(TypedDict):
    """Data type for training and testing data"""

    input1: int
    output1: float


IOdata = list[IOpair]


def print_tree(root: Node, indent: str = "") -> None:
    """
    Pretty-prints the data structure in actual tree form.
    >>> print_tree(root=root, indent="")
    """
    if root.right is not None and root.left is not None:
        print_tree(root=root.right, indent=indent + "    ")
        print(indent, root.data)
        print_tree(root=root.left, indent=indent + "    ")
    else:
        print(indent + root.data)


def parse_expression(original_code: str) -> Node:
    """
    Turns prefix code into a tree data structure.
    >>> clojure_code = "( * ( + 20 45 ) ( - 56 X ) )"
    >>> root = parse_expression(clojure_code)
    """
    original_code = original_code.replace("(", "")
    original_code = original_code.replace(")", "")
    code_arr = original_code.split()
    return _parse_experession(code_arr)


def _parse_experession(code: list[str]) -> Node:
    """
    The back-end helper of parse_expression.
    Not intended for calling directly.
    Assumes code is prefix notation lisp with space delimeters.
    """
    if code[0] in OPERATORS:
        return Node(
            data=code.pop(0),
            left=_parse_experession(code),
            right=_parse_experession(code),
        )
    else:
        return Node(code.pop(0))


def parse_tree_print(root: Node) -> None:
    """
    Stringifies to std-out (print) the tree data structure.
    >>> parse_tree_print(root)
    """
    if root.right is not None and root.left is not None:
        print(f"( {root.data} ", end="")
        parse_tree_print(root.left)
        parse_tree_print(root.right)
        print(") ", end="")
    else:
        # for the case of literal programs... e.g., `4`
        print(f"{root.data} ", end="")


def parse_tree_return(root: Node) -> str:
    """
    Stringifies to the tree data structure, returns string.
    >>> stringified = parse_tree_return(root)
    """
    if root.right is not None and root.left is not None:
        return f"( {root.data} {parse_tree_return(root.left)} {parse_tree_return(root.right)} )"
    else:
        # for the case of literal programs... e.g., `4`
        return root.data


def initialize_individual(genome: str, fitness: float) -> Individual:
    """
    Purpose:        Create one individual
    Parameters:     genome as Node, fitness as integer (higher better)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a dict[Node, int]
    Modifies:       Nothing
    Calls:          Basic python only
    Example doctest:
    >>> ind1 = initialize_individual("( + ( * C ( / 9 5 ) ) 32 )", 0)
    """
    return {"genome": parse_expression(genome), "fitness": fitness}


def initialize_data(input1: int, output1: float) -> IOpair:
    """
    For mypy...
    """
    return {"input1": input1, "output1": output1}


def prefix_to_infix(prefix: str) -> str:
    """
    My minimal lisp on python interpreter, lol...
    >>> C = 0
    >>> print(prefix_to_infix("( + ( * C ( / 9 5 ) ) 32 )"))
    >>> print(eval(prefix_to_infix("( + ( * C ( / 9 5 ) ) 32 )")))
    """
    prefix = prefix.replace("(", "")
    prefix = prefix.replace(")", "")
    prefix_arr = prefix.split()
    stack = []
    i = len(prefix_arr) - 1
    while i >= 0:
        if prefix_arr[i] not in OPERATORS:
            stack.append(prefix_arr[i])
            i -= 1
        else:
            str = "(" + stack.pop() + prefix_arr[i] + stack.pop() + ")"
            stack.append(str)
            i -= 1
    return stack.pop()


def put_an_x_in_it(formula: str) -> str:
    formula = formula.replace("(", "")
    formula = formula.replace(")", "")
    formula_arr = formula.split()
    while True:
        i = random.randint(0, len(formula_arr) - 1)
        if formula_arr[i] not in OPERATORS:
            formula_arr[i] = "x"
            break
    return " ".join(formula_arr)


def gen_rand_prefix_code(depth_limit: int, rec_depth: int = 0) -> str:
    """
    Generates one small formula,
    from OPERATORS and ints from -100 to 200
    """
    rec_depth += 1
    if rec_depth < depth_limit:
        if random.random() < 0.9:
            return (
                random.choice(OPERATORS)
                + " "
                + gen_rand_prefix_code(depth_limit, rec_depth)
                + " "
                + gen_rand_prefix_code(depth_limit, rec_depth)
            )
        else:
            return str(random.randint(-100, 100))
    else:
        return str(random.randint(-100, 100))


# <<<< DO NOT MODIFY


def initialize_pop(pop_size: int) -> Population:
    """
    Purpose:        Create population to evolve
    Parameters:     Goal string, population size as int
    User Input:     no
    Prints:         no
    Returns:        a population, as a list of Individuals
    Modifies:       Nothing
    Calls:          random.choice-1, string.ascii_letters-1, initialize_individual-n
    Example doctest:
    """
    population = []
    i = 0

    # create a random formula. sub the x into the formula. the input from iodata is used to replace x
    # y is the formula as a whole
    # errors are calculated by subtracting y's total from the output. bigger difference the worse it is

    while i < pop_size:
        rand_genome = gen_rand_prefix_code(
            depth_limit=4
        )  # eval_indiv will change from pre to infix
        genome = put_an_x_in_it(rand_genome)

        individual = initialize_individual(genome=genome, fitness=0)
        population.append(individual)
        i += 1

    return population


def inorder_tree_traversal(root: Optional[Node]) -> list[Node]:
    """returns a list of the subtree values with inorder traversal (left root right)"""

    tree_list: list[Node] = []
    inorder_tree_traversal_util(root=root, tree_list=tree_list)

    return tree_list


def inorder_tree_traversal_util(root: Optional[Node], tree_list: list[Node]) -> None:
    """util to recursivly get tree in the inorder traversal - returns nothing"""
    if root is None:
        return

    inorder_tree_traversal_util(root=root.left, tree_list=tree_list)
    tree_list.append(root)
    inorder_tree_traversal_util(root=root.right, tree_list=tree_list)
    return

    # helper functions


def total_node_elements(root: Optional[Node]) -> int:
    """total number of nodes in the genome"""
    if type(root) == type(None):
        return 0
    else:
        total_nodes = (
            total_node_elements(root=root.left)
            + total_node_elements(root=root.right)
            + 1
        )
        return total_nodes


# def child_elements(root: Node | None) -> int:
#     """determine number of children nodes for node selection """
#     count = 0
#     if root != None:
#         count = total_node_elements(root) + 1
#         return count
#     else:
#         return count # count = 0
def random_node_by_index(root: Optional[Node], i: int) -> Optional[Node]:
    """determines node based on index"""
    if root is None:
        return root
    elif i == (left_children := total_node_elements(root.left)):
        return root
    elif i < left_children:
        return random_node_by_index(root.left, i)  # bottom-most subtree
    else:
        right_i = i - total_node_elements(root.left) - 1
        return random_node_by_index(root.right, right_i)  # right subtree


def random_node(root: Node) -> Node:
    """select a random node for mutation etc. im trying 1000x different functions to find one that works"""
    # conditional makes sure there are children to select
    if total_node_elements(root) < 2:
        return root

    node_choice = random.randint(0, total_node_elements(root) - 1)
    # print(node_choice)
    # parse_tree_print(root)
    # print()
    # think about it the same way as the swaps (keep retrieving until you have a value you want)

    # print(f'selected node: {selected_node.data} \n')

    # print()
    selected_node = inorder_tree_traversal(root)[node_choice]
    # if not (selected_node):
    #     raise Exception("none")

    return selected_node


# def different_length_recombo(individual: Node, subtree_chosen: str) -> list[str]:
#     """determines what kind of recombination to do based on what is in the node
#     dont forget you're passing IN THE NODE :)"""
#     indiv_tree_list = inorder_tree_traversal(
#         individual
#     )  # genome is the Node starting from root?

#     if subtree_chosen == "left":
#         subtree = indiv_tree_list[0:3]

#     else:
#         # for some reason indexing with [-3:-1] is not giving the last element
#         # p1_right_subtree = indiv_tree_list[-3:-1]
#         # there is obviously a better solution to this but my eyeballs hurt
#         subtree = indiv_tree_list[-1:]
#         subtree.append(indiv_tree_list[-2])
#         subtree.append(indiv_tree_list[-3])

#     return subtree


def create_subtree_string(subtree: list[str]) -> str:
    """given a subtree list, create the string version in prefix form
    prefix: ( operator integer integer )

    """
    int1, op, int2 = subtree[0], subtree[1], subtree[2]
    subtree_string = "( {} {} {} )".format(op, int2, int1)

    return subtree_string


def insert_substring(
    p1_genome_str: str,
    p2_genome_str: str,
    p1_subtree_string: str,
    p2_subtree_string: str,
) -> tuple[str, str]:
    """insert the prefix subtree string in the correct spot on the opposite parent"""

    new_p1_genome_str = p1_genome_str.replace(p1_subtree_string, p2_subtree_string)
    new_p2_genome_str = p2_genome_str.replace(p2_subtree_string, p1_subtree_string)

    return new_p1_genome_str, new_p2_genome_str


def make_list_from_str(genome_str: str, spaces: str) -> list[str]:
    """return a list from the genome str for iteration"""
    genome_list: list[str] = []
    if spaces == "yes":  # spaces
        for i in range(len(genome_str)):
            genome_list.append(genome_str[i])
        return genome_list
    else:  # nospaces
        genome_list = genome_str.split()
        return genome_list


def make_prefix_genome_from_list(genome_list: list[str]) -> str:
    """use the list from mutate to create new genome str in prefix notation"""
    new_genome_str = ""

    for item in genome_list:
        new_genome_str += item + " "
    new_genome_str = new_genome_str[:-1]

    return new_genome_str


def node_swap_location(original: Node, new_location: Node) -> None:
    """swap nodes between individuals - recombine helper func"""
    # root/data
    original.data, new_location.data = new_location.data, original.data
    # left side
    original.left, new_location.left = new_location.left, original.left
    # right side
    original.right, new_location.right = new_location.right, original.right


def recombine_pair(parent1: Individual, parent2: Individual) -> Population:
    """
    Purpose:        Recombine two parents to produce two children
    Parameters:     Two parents as Individuals
    User Input:     no
    Prints:         no
    Returns:        A population of size 2, the children
    Modifies:       Nothing
    Calls:          Basic python, random.choice-1, initialize_individual-2
    Example doctest:
    """

    """
    subtree_choice = ["left", "right"]

    # genome strings
    p1_genome_str = parse_tree_return(parent1["genome"])
    p2_genome_str = parse_tree_return(parent2["genome"])

    swap_spot = int(random.choice(range(len(p1_genome_str))))

    # if the same length - normal swap? will have to check this
    if len(p1_genome_str) == len(p2_genome_str):
        child1_genome = p1_genome_str[:swap_spot] + p2_genome_str[swap_spot:]
        child2_genome = p2_genome_str[:swap_spot] + p1_genome_str[swap_spot:]

        child1 = initialize_individual(genome=child1_genome, fitness=0)
        child2 = initialize_individual(genome=child2_genome, fitness=0)

        children = [child1, child2]
    elif len(p1_genome_str) == 1 or len(p2_genome_str) == 1:
        # do nothing if one is just an x?
        children = [parent1, parent2]

    # different lengths means hunt for sub trees?
    elif len(p1_genome_str) > len(p2_genome_str) or len(p1_genome_str) < len(
        p2_genome_str
    ):
        subtree_swap = random.choice(subtree_choice)

        # find the correct subtree using inorder traversal of tree
        p1_subtree = different_length_recombo(parent1["genome"], subtree_swap)
        p2_subtree = different_length_recombo(parent2["genome"], subtree_swap)

        # create a string of the subtree to add to new prefix genome string
        p1_subtree_string = create_subtree_string(p1_subtree)
        p2_subtree_string = create_subtree_string(p2_subtree)

        # insert the subtree strings into opposite parents at correct locations
        child1_genome, child2_genome = insert_substring(
            p1_genome_str, p2_genome_str, p1_subtree_string, p2_subtree_string
        )

        # initialize with new prefix genomes
        child1 = initialize_individual(genome=child1_genome, fitness=0)
        child2 = initialize_individual(genome=child2_genome, fitness=0)

        children = [child1, child2]
    """
    children = []
    # all of this wound up being useless because I can easily swap nodes, apparently. I can't bring myself to delete it even though it's ugly :\
    # start with parent's genome
    child1_genome = parse_tree_return(parent1["genome"])
    child2_genome = parse_tree_return(parent2["genome"])

    # initialize them so their nodes are set up
    child1 = initialize_individual(genome=child1_genome, fitness=0)
    child2 = initialize_individual(genome=child2_genome, fitness=0)
    # determine their node swap spots
    # i = random.randint(0,total_node_elements(root))
    child_length = min(
        len(inorder_tree_traversal(child1["genome"])),
        len(inorder_tree_traversal(child2["genome"])),
    )

    # swap more than just one node
    i = 0
    while i < child_length // 2:
        child1_node = random_node(child1["genome"])
        child2_node = random_node(child2["genome"])
        # execute swap
        node_swap_location(original=child1_node, new_location=child2_node)

        i += 1

    children.extend([child1, child2])  # extend population

    return children


def recombine_group(parents: Population, recombine_rate: float) -> Population:
    """
    Purpose:        Recombines a whole group, returns the new population
                    Pair parents 1-2, 2-3, 3-4, etc..
                    Recombine at rate, else clone the parents.
    Parameters:     parents and recombine rate
    User Input:     no
    Prints:         no
    Returns:        New population of children
    Modifies:       Nothing
    Calls:          Basic python, random.random~n/2, recombine pair-n
    """
    children = []

    for pair in range(0, len(parents) - 1, 2):
        if random.random() < recombine_rate:
            child1, child2 = recombine_pair(
                parent1=parents[pair], parent2=parents[pair + 1]
            )
        else:
            child1, child2 = parents[pair], parents[pair + 1]
        children.extend([child1, child2])

    # check to ensure there is an x in the tree/genome string
    checked_children = []
    for item in children:
        genome_str = parse_tree_return(item["genome"])
        if "x" not in genome_str:
            genome_str = put_an_x_in_it(genome_str)
            new_child = initialize_individual(genome=genome_str, fitness=0)
            checked_children.append(new_child)
        else:
            checked_children.append(item)

    return children


def mutate_individual(parent: Individual, mutate_rate: float) -> Individual:
    """
    Purpose:        Mutate one individual
    Parameters:     One parents as Individual, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    Calls:          Basic python, random,choice-1,
    Example doctest:
    """
    # in genetic programming mutation means a random change in the trees
    # randomly selected node to mutate?
    # randomly select operators? integers? percentage based?
    # all ints between (-100,100)
    # random.randint(-100, 100)
    rate = random.random()

    if random.random() < mutate_rate:
        # swap_spot = random.randint(0, len(genome_list) - 1)
        mutant = random_node(parent["genome"])

        # if the node is an operator or integer have a chance to change them accordingly. different variations for diversity
        if mutant.data in OPERATORS:
            if rate < 0.3:  # become an integer
                mutant.data = str(random.randint(-100, 100))
                mutant.left = None
                mutant.right = None
            else:  # become a different operator
                mutant.data = random.choice(OPERATORS)
        else:  # if x, will have to replace x in the genome
            # if mutant.data == 'x':
            if rate < 1:
                # new ints and operator
                new_op = random.choice(OPERATORS)
                new_tree_int1 = str(random.randint(-100, 100))
                new_tree_int2 = str(random.randint(-100, 100))
                # data replaced in the tree
                mutant.data = new_op
                mutant.left = Node(new_tree_int1)
                mutant.right = Node(new_tree_int2)
            else:
                new_int = str(random.randint(-100, 100))
                mutant.data = new_int

        # turn list back into prefix genome string
        # mutant_genome = make_prefix_genome_from_list(genome_list)
        mutant_genome = parse_tree_return(mutant)
        if "x" not in mutant_genome:
            mutant_genome = put_an_x_in_it(mutant_genome)

        new_mutant = initialize_individual(genome=mutant_genome, fitness=0)
        return new_mutant
    else:  # do nothing, return the parent
        return parent


def mutate_group(children: Population, mutate_rate: float) -> Population:
    """
    Purpose:        Mutates a whole Population, returns the mutated group
    Parameters:     Population, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    Calls:          Basic python, mutate_individual-n
    Example doctest:
    """
    mutants = []

    for child in children[1:]:
        mutants.append(mutate_individual(parent=child, mutate_rate=mutate_rate))
    return mutants


# DO NOT MODIFY >>>>
def evaluate_individual(individual: Individual, io_data: IOdata) -> None:
    """
    Purpose:        Computes and modifies the fitness for one individual
    Parameters:     One Individual, data formatted as IOdata
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The individual (mutable object)
    Calls:          Basic python only
    Notes:          train/test format is like PSB2 (see IOdata above)
    Example doctest:
    >>> evaluate_individual(ind1, io_data)
    """
    fitness = 0
    errors = []
    for sub_eval in io_data:
        eval_string = parse_tree_return(individual["genome"]).replace(
            "x", str(sub_eval["input1"])
        )

        # In clojure, this is really slow with subprocess
        # eval_string = "( float " + eval_string + ")"
        # returnobject = subprocess.run(
        #     ["clojure", "-e", eval_string], capture_output=True
        # )
        # result = float(returnobject.stdout.decode().strip())

        # In python, this is MUCH MUCH faster:
        try:
            y = eval(prefix_to_infix(eval_string))
        except ZeroDivisionError:
            y = math.inf

        errors.append(abs(sub_eval["output1"] - y))
    # Higher errors is bad, and longer strings is bad
    fitness = sum(errors) + len(eval_string.split())
    # Higher fitness is worse
    individual["fitness"] = fitness


# <<<< DO NOT MODIFY


def evaluate_group(individuals: Population, io_data: IOdata) -> None:
    """
    Purpose:        Computes and modifies the fitness for population
    Parameters:     Objective string, Population
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The Individuals, all mutable objects
    Calls:          Basic python, evaluate_individual-n
    Example doctest:
    """
    for indiv_index in range(len(individuals)):
        evaluate_individual(individuals[indiv_index], io_data)

    # individuals[:] = [x for x in individuals if not x["fitness"]== math.inf] # remove infinite fitness from selection


def rank_group(individuals: Population) -> None:
    """
    Purpose:        Create one individual
    Parameters:     Population of Individuals
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The population's order (a mutable object)
    Calls:          Basic python only
    Example doctest:
    """
    individuals.sort(key=lambda ind: ind["fitness"], reverse=False)


def parent_select(individuals: Population, number: int) -> Population:
    """
    Purpose:        Choose parents in direct probability to their fitness
    Parameters:     Population, the number of individuals to pick.
    User Input:     no
    Prints:         no
    Returns:        Sub-population
    Modifies:       Nothing
    Calls:          Basic python, random.choices-1
    Example doctest:
    """

    # how should I deal with individual nodes
    sub_population: list[Individual] = []

    rank_group(individuals)
    sub_population[:] = [
        x for x in individuals if not x["fitness"] == math.inf
    ]  # remove infinite fitness from selection

    return sub_population


def survivor_select(individuals: Population, pop_size: int) -> Population:
    """
    Purpose:        Picks who gets to live!
    Parameters:     Population, and population size to return.
    User Input:     no
    Prints:         no
    Returns:        Population, of pop_size
    Modifies:       Nothing
    Calls:          Basic python only
    Example doctest:
    """

    og_survivor_pop, new_survivor_pop = [], []

    # never touch the best one
    elitism = pop_size // 5
    indiv_size = len(individuals)
    i = 0
    while i < pop_size - elitism:  # subtract our best one
        og_survivor_pop.append(
            individuals[random.randrange(0, indiv_size - 1)]
        )  # can give repeats
        i += 1
    for i in range(elitism):
        og_survivor_pop.append(individuals[i])

    rank_group(og_survivor_pop)
    for item in og_survivor_pop:
        if item["fitness"] != math.inf:
            new_survivor_pop.append(item)

    return new_survivor_pop


def evolve(io_data: IOdata, pop_size: int = 500) -> Population:
    """
    Purpose:        A whole EC run, main driver
    Parameters:     The evolved population of solutions
    User Input:     No
    Prints:         Updates every time fitness switches.
    Returns:        Population
    Modifies:       Various data structures
    Calls:          Basic python, all your functions
    """
    # To debug doctest test in pudb
    # Highlight the line of code below below
    # Type 't' to jump 'to' it
    # Type 's' to 'step' deeper
    # Type 'n' to 'next' over
    # Type 'f' or 'r' to finish/return a function call and go back to caller

    # initial fitnesses
    fitnesses, epochs = [], []
    plot = True

    counter = 0
    population = initialize_pop(pop_size)
    evaluate_group(population, io_data)
    rank_group(population)
    best_fitness = population[0]["fitness"]
    print()
    print(best_fitness)
    best = copy.deepcopy(population[0])

    # for _ in tqdm(range(500)):
    # for _ in range(500):
    while best_fitness > 40 and counter < 3000:
        counter += 1
        parents = parent_select(individuals=population, number=int(pop_size * 0.65))
        children = recombine_group(parents=parents, recombine_rate=0.4)

        # cyclical - nestling in :)
        if counter < 1000:
            mutants = mutate_group(children=children, mutate_rate=0.7)
        elif counter > 1000 and counter < 2000:
            mutants = mutate_group(children=children, mutate_rate=0.9)
        else:
            mutants = mutate_group(children=children, mutate_rate=0.5)
        everyone = mutants
        everyone.append(best)
        evaluate_group(everyone, io_data)
        rank_group(everyone)
        best = copy.deepcopy(everyone[0])

        population = survivor_select(individuals=everyone, pop_size=pop_size)
        best_fitness = population[0]["fitness"]

        if not counter % 1:
            print()
            print(best_fitness)
            print(counter)
            total = 0
            # for item in population:
            #     total += item["fitness"]

            fitnesses.append(best_fitness)
            epochs.append(counter)
        if counter == 999:
            print()
    if plot:
        plt.plot(epochs, fitnesses, "ro")
        plt.ylim(0, 20000)
        # plt.savefig("./test.png")
        plt.show()
    population.insert(0, best)
    return population


# Seed for base grade.
# For the exploratory competition points (last 10),
# comment this one line out if you want, but put it back please.
seed = False

# DO NOT MODIFY >>>>
if __name__ == "__main__":
    divider = "===================================================="
    # Execute doctests to protect main:
    # import doctest

    # doctest.testmod()
    # doctest.testmod(verbose=True)

    if seed:
        random.seed(42)

    print(divider)
    print("Number of possible genetic programs: infinite...")
    print("Lower fitness is better.")
    print(divider)

    X = list(range(-10, 110, 10))
    Y = [(x * (9 / 5)) + 32 for x in X]
    # data = [{"input1": x, "output1": y} for x, y in zip(X, Y)]
    # mypy wanted this:
    counter = 1
    while True:
        print("======================================= Test", counter)
        rand_code = gen_rand_prefix_code(depth_limit=4)
        rand_code = put_an_x_in_it(rand_code)
        print("Prefix code:")
        parse_tree_print(parse_expression(rand_code))
        test_code = prefix_to_infix(rand_code)
        print(f"\nInfix code:\n{test_code}")
        X = random.choices(range(-100, 100), k=10)
        try:
            Y = [eval(test_code) for x in X]
            counter += 1
            break
        except ZeroDivisionError:
            pass

    data = [{"input1": x, "output1": y} for x, y in zip(X, Y)]

    data = [initialize_data(input1=x, output1=y) for x, y in zip(X, Y)]

    # Correct:
    print("Example of celcius to fahrenheit:")
    ind1 = initialize_individual("( + ( * x ( / 9 5 ) ) 32 )", 0)
    evaluate_individual(ind1, data)
    print_tree(ind1["genome"])
    print("Fitness", ind1["fitness"])

    # Yours
    train = data[: int(len(data) / 2)]
    test = data[int(len(data) / 2) :]
    population = evolve(train)
    evaluate_individual(population[0], test)
    population[0]["fitness"]

    print("Here is the best program:")
    parse_tree_print(population[0]["genome"])
    print("And it's fitness:")
    print(population[0]["fitness"])
# <<<< DO NOT MODIFY

# %%
