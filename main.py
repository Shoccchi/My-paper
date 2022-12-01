from __future__ import annotations

from typing import TypeVar, List, Dict
from random import choices, random, randrange, shuffle
from heapq import nlargest
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime


class Chromosome(ABC):
    """
    An abstract class that deals with chromosomes (one element of a genetic algorithm).
    """

    @abstractmethod
    def get_fitness(self) -> float:
        """
        Abstract method for evaluation function Y to obtain the chromosome excellence for the target problem.

        Returns
        -------
        fitness : float
            The value of the chromosome's excellence for the problem in question. The higher the value, the more suitable the chromosome is for the problem. It is also used to determine the termination of genetic algorithms.
        """
        ...

    @classmethod
    @abstractmethod
    def make_random_instance(cls) -> Chromosome:
        """
        Create instances with random features (attribute values) Abstract Method

        Returns
        -------
        instance : Chromosome
            Generated instances
        """
        ...

    @abstractmethod
    def mutate(self) -> None:
        """
        Abstract method of processing to (mutate) chromosomes.
        The setting of random alternate values for instance attributes, etc., is performed.
        """
        ...

    @abstractmethod
    def exec_crossover(self, other: Chromosome) -> List[Chromosome]:
        """
        The crossings are performed by referring to another individual specified in the argument.

        Parameters
        ----------
        other : Chromosome
            Another individual used in crossings

        Returns
        -------
        result_chromosomes : list of Chromosome
            Two individuals (chromosomes) generated after a crossover run
        """
        ...

    def __lt__(self, other: Chromosome) -> bool:
        """
        A function for comparing smaller values of the evaluation function, used in comparisons between individuals

        Parameters
        ----------
        other : Chromosome
            Other individuals for comparison

        Returns
        -------
        result_bool : bool
            True/false value of whether or not the condition is satisfied to a small degree
        """
        return self.get_fitness() < other.get_fitness()


C = TypeVar('C', bound=Chromosome)


class GeneticAlgorithm:
    SelectionType = int
    SELECTION_TYPE_ROULETTE_WHEEL: SelectionType = 1
    SELECTION_TYPE_TOURNAMENT: SelectionType = 2

    def __init__(
            self, initial_population: List[C],
            threshold: float,
            max_generations: int, mutation_probability: float,
            crossover_probability: float,
            selection_type: SelectionType) -> None:
        """
        Class for genetic algorithms

        Parameters
        ----------
        initial_population : list of Chromosome
            First generation populations (chromosome groups)
        threshold : float
            Thresholds used in problem-solving decisions
            The calculation is terminated when an individual exceeds this value.
        max_generations : int
            Maximum number of generations to be performed by the algorithm
        mutation_probability : float
            Mutation probability（0.0～1.0）。
        crossover_probability : float
            Crossing probability（0.0～1.0）。
        selection_type : int
            Selection Method. Specify one of the following constant values.
            - SELECTION_TYPE_ROULETTE_WHEEL
            - SELECTION_TYPE_TOURNAMENT
        """
        self._population: List[Chromosome] = initial_population
        self._threshold: float = threshold
        self._max_generations: int = max_generations
        self._mutation_probability: float = mutation_probability
        self._crossover_probability: float = crossover_probability
        self._selection_type: int = selection_type

    def _exec_roulette_wheel_selection(self) -> List[Chromosome]:
        """
        Roulette selection is performed to obtain two individuals (chromosomes) to be used for mating, etc.



        Returns
        -------
        selected_chromosomes : list of Chromosome
            A list containing two selected individuals (chromosomes)
            The selection process is randomly extracted with the weights set by the evaluation function (fitness method).

        Notes
        -----
        It cannot be used for problems where the resulting value of the evaluation function is negative.
        """
        weights: List[float] = [
            chromosome.get_fitness() for chromosome in self._population]
        selected_chromosomes: List[Chromosome] = choices(
            self._population, weights=weights, k=2)
        return selected_chromosomes

    def _exec_tournament_selection(self) -> List[Chromosome]:
        """
        Tournament selection is performed to obtain two individuals (chromosomes) for use in mating, etc.


        Returns
        -------
        selected_chromosomes : list of Chromosome
            List containing two selected individuals (chromosomes)
            The top two individuals are set for the tournament from among the number of individuals extracted as specified by the argument.

        """
        participants_num: int = len(self._population) // 2
        participants: List[Chromosome] = choices(self._population, k=participants_num)
        selected_chromosomes: List[Chromosome] = nlargest(n=2, iterable=participants)
        return selected_chromosomes

    def _to_next_generation(self) -> None:
        """
        Generate the next generation of individuals (chromosomes) and replace the attribute values of the population with those of the generated next generation.

        """
        new_population: List[Chromosome] = []

        # The comparison of the number of cases is judged on the basis of a small number of cases rather than equality, taking into account the case where the number of cases in the original population is an odd number.
        while len(new_population) < len(self._population):
            parents: List[Chromosome] = self._get_parents_by_selection_type()
            next_generation_chromosomes: List[Chromosome] = \
                self._get_next_generation_chromosomes(parents=parents)
            new_population.extend(next_generation_chromosomes)

        # 2件ずつ次世代のリストを増やしていく都合、元のリストよりも件数が
        # 多い場合は1件リストから取り除いてリストの件数を元のリストと一致させる。
        if len(new_population) > len(self._population):
            del new_population[0]

        self._population = new_population

    def _get_next_generation_chromosomes(
            self, parents: List[Chromosome]) -> List[Chromosome]:
        """
        From the calculated list of two individuals of each parent,
        Obtain a list of two populations to treat as the next generation.
        It can be crossed or mutated with a certain probability, and
        if the probability is not satisfied, the value of the argument is set as the next generation as it is.

        Parameters
        ----------
        parents : list of Chromosome
            List of two individuals of each parent calculated

        Returns
        -------
        next_generation_chromosomes : list of Chromosome
            A list containing two individuals, set up as the next generation.
        """
        random_val: float = random()
        next_generation_chromosomes: List[Chromosome] = parents
        if random_val < self._crossover_probability:
            next_generation_chromosomes = parents[0].exec_crossover(
                other=parents[1])

        random_val = random()
        if random_val < self._mutation_probability:
            for chromosome in next_generation_chromosomes:
                chromosome.mutate()
        return next_generation_chromosomes

    def _get_parents_by_selection_type(self) -> List[Chromosome]:
        """
        Obtain a list of the two individuals (chromosomes) of the parent according to the selection method.

        Returns
        -------
        parents : list of Chromosome
            List of two individuals (chromosomes) of the acquired parents.

        Raises
        ------
        ValueError
            If an unsupported selection method is specified.
        """
        if self._selection_type == self.SELECTION_TYPE_ROULETTE_WHEEL:
            parents: List[Chromosome] = self._exec_roulette_wheel_selection()
        elif self._selection_type == self.SELECTION_TYPE_TOURNAMENT:
            parents = self._exec_tournament_selection()
        else:
            raise ValueError(
                'An unsupported selection method has been specified. : %s'
                % self._selection_type)
        return parents

    def run_algorithm(self) -> Chromosome:
        """
        Execute the genetic algorithm and obtain an instance of the individual (chromosome) resulting from the execution of the result of the execution.

        Returns
        -------
        betst_chromosome : Chromosome
            Individuals as a result of algorithm execution.
            If the threshold is exceeded by the evaluation function, or if the threshold is not exceeded, the individual with the highest value of the evaluation function is set when the specified number of generations is reached.
        """
        best_chromosome: Chromosome = \
            deepcopy(self._get_best_chromosome_from_population())
        for generation_idx in range(self._max_generations):
            print(
                datetime.now(),
                f' Number of generation : {generation_idx}'
                f' Best individual : {best_chromosome}'
            )

            if best_chromosome.get_fitness() >= self._threshold:
                return best_chromosome

            self._to_next_generation()

            currrent_generation_best_chromosome: Chromosome = \
                self._get_best_chromosome_from_population()
            current_gen_best_fitness: float = \
                currrent_generation_best_chromosome.get_fitness()
            if best_chromosome.get_fitness() < current_gen_best_fitness:
                best_chromosome = deepcopy(currrent_generation_best_chromosome)
        return best_chromosome

    def _get_best_chromosome_from_population(self) -> Chromosome:
        """
        Obtain the individual (chromosome) with the highest evaluation function value from the list of populations.


        Returns
        -------
        best_chromosome : Chromosome
            The individual with the highest value of the evaluation function in the list.
        """
        best_chromosome: Chromosome = self._population[0]
        for chromosome in self._population:
            if best_chromosome.get_fitness() < chromosome.get_fitness():
                best_chromosome = chromosome
        return best_chromosome


class SimpleEquationProblem(Chromosome):

    def __init__(self, x: int, y: int) -> None:
        """
        This class deals with the problem of finding the value of x and y that maximizes the value of the following simple equation for checking the operation of a genetic algorithm.
        6x - x^2 + 4 * y - y^2

        （Answer is x = 3, y = 2）

        Parameters
        ----------
        x : int
            Initial value of x
        y : int
            Initial value of y
        """
        self.x = x
        self.y = y

    def get_fitness(self) -> float:
        """
        6x - x^2 + 4 * y - y^2
        A method used as an evaluation function to obtain the value of the result of the calculation of the above expression by the current x and y values.

        Returns
        -------
        fitness : int
        The value of the result of the formula calculation.

        """
        x: int = self.x
        y: int = self.y
        return 6 * x - x ** 2 + 4 * y - y ** 2

    @classmethod
    def make_random_instance(cls) -> SimpleEquationProblem:
        """
        Create an instance of the SimpleEquationProblem class with random initial values.

        Returns
        -------
        problem : SimpleEquationProblem
            Generated instances.
            x and y are set to random values in the range 0 to 99.
        """
        x: int = randrange(100)
        y: int = randrange(100)
        problem = SimpleEquationProblem(x=x, y=y)
        return problem

    def mutate(self) -> None:
        """
        Mutate (mutate) an individual.
        (Increase or decrease the value of x or y by 1, depending on the random number).
        """
        value: int = choices([1, -1], k=1)[0]
        if random() > 0.5:
            self.x += value
            return
        self.y += value

    def exec_crossover(
            self, other: SimpleEquationProblem
    ) -> List[SimpleEquationProblem]:
        """
        The crossings are performed by referring to another individual specified in the argument.


        Parameters
        ----------
        other : SimpleEquationProblem
            Another individual used in crossings.

        Returns
        -------
        result_chromosomes : list of SimpleEquationProblem
            A list containing the two individuals generated after the crossover run.
            The individual inherits half of the x and y values from each of the parent individuals.
        """
        child_1: SimpleEquationProblem = deepcopy(self)
        child_2: SimpleEquationProblem = deepcopy(other)
        child_1.y = other.y
        child_2.x = self.x
        result_chromosomes: List[SimpleEquationProblem] = [
            child_1, child_2,
        ]
        return result_chromosomes

    def __str__(self) -> str:
        """
        Returns a string of individual information.

        Returns
        -------
        info : str
            Individual information string
        """
        x: int = self.x
        y: int = self.y
        fitness: float = self.get_fitness()
        info: str = f'x = {x}, y = {y}, fitness = {fitness}'
        return info


LetterDict = Dict[str, int]


class SendMoreMoneyProblem(Chromosome):
    LETTERS: List[str] = ['S', 'E', 'N', 'D', 'M', 'O', 'R', 'Y']

    def __init__(self, letters_dict: LetterDict) -> None:
        """
        SEND + MORE = MONEY
        This class is designed to solve this overhead arithmetic problem with a genetic algorithm.

        Parameters
        ----------
        letters_dict : LetterDict
            A dictionary that stores the initial values of each of the 8 characters (keys) used in the problem and the numerical values assigned to them.
        """
        self.letters_dict: LetterDict = letters_dict

    def get_fitness(self) -> float:
        """
        SEND + MORE values by the current numerical value assigned to each character and Method for evaluation function based on the difference between the numerical value of MONEY and the numerical value of SEND + MORE by the numerical value assigned to each current character.

        Notes
        -----
        Since the value of the evaluation function of the genetic algorithm is a highly valued form,
        the value is returned with the value adjusted so that the larger the error, the lower the value.


        Returns
        -------
        fitness : int
            Valuation value based on the difference between the SEND + MORE value and the MONEY value.
            The smaller the difference, the larger the value.
        """
        send_val: int = self._get_send_val()
        more_val: int = self._get_more_val()
        money_val: int = self._get_money_val()
        difference: int = abs(money_val - (send_val + more_val))
        return 1 / (difference + 1)

    def _get_send_val(self) -> int:
        """
        Obtains the value of SEND by the numerical value of each character currently assigned.

        Returns
        -------
        send_val : int
            The value of SEND by the numerical value of each character currently assigned.
        """
        s: int = self.letters_dict['S']
        e: int = self.letters_dict['E']
        n: int = self.letters_dict['N']
        d: int = self.letters_dict['D']
        send_val: int = s * 1000 + e * 100 + n * 10 + d
        return send_val

    def _get_more_val(self) -> int:
        """
        Obtains the value of MORE by the numerical value of each character currently assigned.

        Parameters
        ----------
        more_val : int
            MORE value by the numerical value of each character currently assigned
        """
        m: int = self.letters_dict['M']
        o: int = self.letters_dict['O']
        r: int = self.letters_dict['R']
        e: int = self.letters_dict['E']
        more_val: int = m * 1000 + o * 100 + r * 10 + e
        return more_val

    def _get_money_val(self):
        """
        Obtain the value of MONEY by the numerical value of each character currently assigned.

        Returns
        -------
        money_val : int
            The value of MONEY by the numerical value of each character currently assigned.
        """
        m: int = self.letters_dict['M']
        o: int = self.letters_dict['O']
        n: int = self.letters_dict['N']
        e: int = self.letters_dict['E']
        y: int = self.letters_dict['Y']
        money_val = m * 10000 + o * 1000 + n * 100 + e * 10 + y
        return money_val

    @classmethod
    def make_random_instance(cls) -> SendMoreMoneyProblem:
        """
        Create an instance of the SendMoreMoneyProblem class given a random initial value. instance of the SendMoreMoneyProblem class with random initial values.

        Returns
        -------
        problem : SendMoreMoneyProblem
            Generated instances.
            Each character is assigned a value in the range of 0 to 9 with no duplicate numbers.
        """
        num_list: List[int] = list(range(10))
        shuffle(num_list)
        num_list = num_list[:len(cls.LETTERS)]
        letters_dict: LetterDict = {
            char: num for (char, num) in zip(cls.LETTERS, num_list)}

        problem: SendMoreMoneyProblem = SendMoreMoneyProblem(
            letters_dict=letters_dict)
        return problem

    def mutate(self) -> None:
        """
        Mutate (mutate) an individual (randomly replace the value of a particular letter with an unassigned value). (randomly replacing the value of a particular letter with an unassigned value).
        """
        target_char: str = choices(self.LETTERS, k=1)[0]
        not_assigned_num: int = self._get_not_assigned_num()
        self.letters_dict[target_char] = not_assigned_num

    def _get_not_assigned_num(self) -> int:
        """
        Obtain a number that is not assigned to each character.

        Returns
        -------
        not_assigned_num : int
            Numbers not assigned to each letter
            While there are 8 letters, there are 10 numbers (0-9), so there are 2 unassigned numbers, one of which is set.

        """
        values: list = list(self.letters_dict.values())
        not_assigned_num: int = -1
        for num in range(10):
            if num in values:
                continue
            not_assigned_num = num
            break
        return not_assigned_num

    def exec_crossover(
            self,
            other: SendMoreMoneyProblem) -> List[SendMoreMoneyProblem]:
        """
        The crossings are performed by referring to another individual specified in the argument.

        Parameters
        ----------
        other : SendMoreMoneyProblem
            Another individual used in crossings.

        Returns
        -------
        result_chromosomes : list of SendMoreMoneyProblem
            A list containing the two individuals generated after the crossover run.
        """
        child_1: SendMoreMoneyProblem = deepcopy(self)
        child_2: SendMoreMoneyProblem = deepcopy(other)

        for char in ('S', 'E', 'N', 'D'):
            child_2.letters_dict[char] = self. \
                _get_not_assigned_num_from_parent(
                child=child_2,
                parent=self,
            )
        for char in ('M', 'O', 'R', 'Y'):
            child_1.letters_dict[char] = \
                self._get_not_assigned_num_from_parent(
                    child=child_1,
                    parent=other,
                )

        result_chromosomes = [child_1, child_2]
        return result_chromosomes

    def _get_not_assigned_num_from_parent(
            self, child: SendMoreMoneyProblem,
            parent: SendMoreMoneyProblem) -> int:
        """
        Obtain a number that has been set for
        the parent that has not yet been set for the child.

        Notes
        -----
        In some cases, depending on the combination of parent and child values,
        a value that can be selected may not be found,
        in which case a value not assigned among the values 0 to 9 is set.

        Parameters
        ----------
        child : SendMoreMoneyProblem
            Individuals of child
        parent : SendMoreMoneyProblem
            Parent individuals

        Returns
        -------
        not_assigned_num : int
            A number that has not yet been assigned that has been calculated
        """
        not_assigned_num: int = -1
        for parent_num in parent.letters_dict.values():
            child_nums: list = list(child.letters_dict.values())
            if parent_num in child_nums:
                continue
            not_assigned_num = parent_num
        if not_assigned_num == -1:
            not_assigned_num = self._get_not_assigned_num()
        return not_assigned_num

    def __str__(self) -> str:
        """
        Returns a string of individual information.

        Returns
        -------
        info : str
            Individual information string.
        """
        send_val: int = self._get_send_val()
        more_val: int = self._get_more_val()
        money_val: int = self._get_money_val()
        difference: int = abs(money_val - (send_val + more_val))
        info: str = (
            f"\nS = {self.letters_dict['S']}"
            f"　E = {self.letters_dict['E']}"
            f"　N = {self.letters_dict['N']}"
            f"　D = {self.letters_dict['D']}"
            f"\nM = {self.letters_dict['M']}"
            f"　O = {self.letters_dict['O']}"
            f"　R = {self.letters_dict['R']}"
            f"　Y = {self.letters_dict['Y']}"
            f'\nSEND = {send_val}'
            f'　MORE = {more_val}'
            f'　MONEY = {money_val}'
            f'　difference : {difference}'
            '\n--------------------------------'
        )
        return info


if __name__ == '__main__':
    simple_equation_initial_population: List[SimpleEquationProblem] = \
        [SimpleEquationProblem.make_random_instance() for _ in range(30)]
    ga: GeneticAlgorithm = GeneticAlgorithm(
        initial_population=simple_equation_initial_population,
        threshold=13,
        max_generations=100,
        mutation_probability=0.2,
        crossover_probability=0.3,
        selection_type=GeneticAlgorithm.SELECTION_TYPE_TOURNAMENT)
    _ = ga.run_algorithm()

    send_more_money_initial_population: List[SendMoreMoneyProblem] = \
        [SendMoreMoneyProblem.make_random_instance() for _ in range(1000)]
    ga = GeneticAlgorithm(
        initial_population=send_more_money_initial_population,
        threshold=1.0,
        max_generations=1000,
        mutation_probability=0.7,
        crossover_probability=0.2,
        selection_type=GeneticAlgorithm.SELECTION_TYPE_ROULETTE_WHEEL)
    _ = ga.run_algorithm()
