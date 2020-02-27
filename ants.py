"""CS 61A presents Ants Vs. SomeBees."""

import random
from ucb import main, interact, trace
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import copy

################
# Core Classes #
################

class Place(object):
    """A Place holds insects and has an exit to another Place."""

    def __init__(self, name, exit=None):
        """Create a Place with the given NAME and EXIT.

        name -- A string; the name of this Place.
        exit -- The Place reached by exiting this Place (may be None).
        """
        self.name = name
        self.exit = exit
        self.bees = []        # A list of Bees
        self.ant = None       # An Ant
        self.entrance = None  # A Place
        if(self.exit):
            self.exit.entrance = self
        self.location = np.zeros([2],dtype=int)
        if(name[0:6]=='tunnel'):
            self.location[0] = int(self.name[7])
            self.location[1] = int(self.name[9])

        # 'tunnel_0_0'
        # Phase 1: Add an entrance to the exit
        # BEGIN Problem 2
        # "*** YOUR CODE HERE ***"
        # END Problem 2

    def add_insect(self, insect):
        """Add an Insect to this Place.

        There can be at most one Ant in a Place, unless exactly one of them is
        a container ant (Problem 9), in which case there can be two. If add_insect
        tries to add more Ants than is allowed, an assertion error is raised.

        There can be any number of Bees in a Place.
        """
        if insect.is_ant:
            if self.ant is None or self.ant.armor<=0:
                self.ant = insect
            else:
                # BEGIN Problem 9
                if(self.ant.is_container == True):
                    if(self.ant.can_contain(insect)):
                        self.ant.contain_ant(insect)
                elif(insect.is_container == True):
                    if(insect.can_contain(self.ant)):
                        insect.contain_ant(self.ant)
                        self.ant = insect
                else:
                    assert self.ant is None, 'Two ants in {0}'.format(self)
                # END Problem 9
        else:
            self.bees.append(insect)
        insect.place = self

    def remove_insect(self, insect):
        """Remove an INSECT from this Place.

        A target Ant may either be directly in the Place, or be contained by a
        container Ant at this place. The true QueenAnt may not be removed. If
        remove_insect tries to remove an Ant that is not anywhere in this
        Place, an AssertionError is raised.

        A Bee is just removed from the list of Bees.
        """
        if insect.is_ant:
            # Special handling for QueenAnt
            # BEGIN Problem 13
            "*** YOUR CODE HERE ***"
            # END Problem 13

            # Special handling for container ants
            if self.ant is insect:
                # Bodyguard was removed. Contained ant should remain in the game
                if hasattr(self.ant, 'is_container') and self.ant.is_container:
                    self.ant = self.ant.contained_ant
                else:
                    self.ant = None
            else:
                # Contained ant was removed. Bodyguard should remain
                if hasattr(self.ant, 'is_container') and self.ant.is_container \
                        and self.ant.contained_ant is insect:
                    self.ant.contained_ant = None
                else:
                    assert False, '{0} is not in {1}'.format(insect, self)
        else:
            self.bees.remove(insect)

        insect.place = None

    def __str__(self):
        return self.name


class Insect(object):
    """An Insect, the base class of Ant and Bee, has armor and a Place."""

    is_ant = False
    damage = 0
    # ADD CLASS ATTRIBUTES HERE

    def __init__(self, armor, place=None):
        """Create an Insect with an ARMOR amount and a starting PLACE."""
        self.armor = armor
        self.place = place  # set by Place.add_insect and Place.remove_insect

    def reduce_armor(self, amount):
        """Reduce armor by AMOUNT, and remove the insect from its place if it
        has no armor remaining.

        >>> test_insect = Insect(5)
        >>> test_insect.reduce_armor(2)
        >>> test_insect.armor
        3
        """
        self.armor -= amount
        if self.armor <= 0:
            self.place.remove_insect(self)
            self.death_callback()

    def action(self, colony):
        """The action performed each turn.

        colony -- The AntColony, used to access game state information.
        """

    def death_callback(self):
        # overriden by the gui
        pass

    def __repr__(self):
        cname = type(self).__name__
        return '{0}({1}, {2})'.format(cname, self.armor, self.place)


class Bee(Insect):
    """A Bee moves from place to place, following exits and stinging ants."""

    name = 'Bee'
    damage = 1
    # OVERRIDE CLASS ATTRIBUTES HERE


    def sting(self, ant):
        """Attack an ANT, reducing its armor by 1."""
        ant.reduce_armor(self.damage)

    def move_to(self, place):
        """Move from the Bee's current Place to a new PLACE."""
        self.place.remove_insect(self)
        place.add_insect(self)

    def blocked(self):
        """Return True if this Bee cannot advance to the next Place."""
        # Phase 4: Special handling for NinjaAnt
        # BEGIN Problem 7
        # return self.place.ant is not None
        if self.place.ant != None:
            if self.place.ant.blocks_path:
                return True
            else:
                return False
        return False
        # END Problem 7

    def action(self, colony):
        """A Bee's action stings the Ant that blocks its exit if it is blocked,
        or moves to the exit of its current place otherwise.

        colony -- The AntColony, used to access game state information.
        """
        destination = self.place.exit
        # Extra credit: Special handling for bee direction
        # BEGIN EC
        "*** YOUR CODE HERE ***"
        # END EC
        if self.blocked():
            self.sting(self.place.ant)
        elif self.armor > 0 and destination is not None:
            self.move_to(destination)


class Ant(Insect):
    """An Ant occupies a place and does work for the colony."""

    is_ant = True
    implemented = False  # Only implemented Ant classes should be instantiated
    food_cost = 0
    blocks_path = True
    is_container = False
    index = 0
    # ADD CLASS ATTRIBUTES HERE

    def __init__(self, armor=1):
        """Create an Ant with an ARMOR quantity."""
        Insect.__init__(self, armor)


    def can_contain(self, other):
        return False

    def action(self, colony):
        pass


class HarvesterAnt(Ant):
    """HarvesterAnt produces 1 additional food per turn for the colony."""

    name = 'Harvester'
    implemented = True
    food_cost = 2
    index = 1
    # OVERRIDE CLASS ATTRIBUTES HERE

    def action(self, colony):
        """Produce 1 additional food for the COLONY.

        colony -- The AntColony, used to access game state information.
        """
        # BEGIN Problem 1
        "*** YOUR CODE HERE ***"
        colony.food += 1
        # END Problem 1


class ThrowerAnt(Ant):
    """ThrowerAnt throws a leaf each turn at the nearest Bee in its range."""

    name = 'Thrower'
    implemented = True
    damage = 1
    food_cost = 3
    min_range, max_range = 0, 10
    index = 2
    # ADD/OVERRIDE CLASS ATTRIBUTES HERE

    def nearest_bee(self, beehive):
        """Return the nearest Bee in a Place that is not the HIVE, connected to
        the ThrowerAnt's Place by following entrances.

        This method returns None if there is no such Bee (or none in range).
        """
        # BEGIN Problem 3 and 4
        loc = self.place
        distance = 0

        while loc.name != 'Hive':
            if random_or_none(loc.bees) == None:
                loc = loc.entrance
                distance +=1
            else:
                if distance >= self.min_range and distance <= self.max_range:
                    return random_or_none(loc.bees)
                else:
                    loc = loc.entrance
                    distance += 1
        return None
        # END Problem 3 and 4

    def throw_at(self, target):
        """Throw a leaf at the TARGET Bee, reducing its armor."""
        if target is not None:
            target.reduce_armor(self.damage)

    def action(self, colony):
        """Throw a leaf at the nearest Bee in range."""
        self.throw_at(self.nearest_bee(colony.beehive))

def random_or_none(s):
    """Return a random element of sequence S, or return None if S is empty."""
    assert isinstance(s, list), "random_or_none's argument should be a list but was a %s" % type(s).__name__
    if s:
        return random.choice(s)

##############
# Extensions #
##############

class ShortThrower(ThrowerAnt):
    """A ThrowerAnt that only throws leaves at Bees at most 3 places away."""

    name = 'Short'
    # OVERRIDE CLASS ATTRIBUTES HERE
    # BEGIN Problem 4
    # implemented = True
    implemented = False
    min_range = 0
    max_range = 2
    food_cost = 2
    index = 3

class LongThrower(ThrowerAnt):
    """A ThrowerAnt that only throws leaves at Bees at least 5 places away."""

    name = 'Long'
    # OVERRIDE CLASS ATTRIBUTES HERE
    # BEGIN Problem 4
    # implemented = True
    implemented = False
    min_range = 4
    max_range = 10
    food_cost = 2
    index = 4
    # END Problem 4

class FireAnt(Ant):
    """FireAnt cooks any Bee in its Place when it expires."""

    name = 'Fire'
    damage = 3
    # OVERRIDE CLASS ATTRIBUTES HERE
    # BEGIN Problem 5
    # implemented = True
    implemented = False
    food_cost = 5
    index = 5
    # END Problem 5

    def __init__(self, armor=3):
        """Create an Ant with an ARMOR quantity."""
        Ant.__init__(self, armor)

    def reduce_armor(self, amount):
        """Reduce armor by AMOUNT, and remove the FireAnt from its place if it
        has no armor remaining.

        Make sure to damage each bee in the current place, and apply the bonus
        if the fire ant dies.
        """
        # BEGIN Problem 5
        "*** YOUR CODE HERE ***"
        self.armor -= amount
        if self.armor <= 0:
            bees = self.place.bees[0:]
            for bee in bees:
                bee.reduce_armor(self.damage)
            # print('{0} ran out of armor and expired'.format(self))
            self.place.remove_insect(self)
        else:
            bees = self.place.bees[0:]
            for bee in bees:
                bee.reduce_armor(amount)
        # END Problem 5

class HungryAnt(Ant):
    """HungryAnt will take three turns to digest a Bee in its place.
    While digesting, the HungryAnt can't eat another Bee.
    """
    name = 'Hungry'
    # OVERRIDE CLASS ATTRIBUTES HERE
    # BEGIN Problem 6
    # implemented = True
    implemented = False
    food_cost = 4 # Change to True to view in the GUI
    index = 6
    # END Problem 6

    def __init__(self, armor=1):
        # BEGIN Problem 6
        "*** YOUR CODE HERE ***"
        self.digesting = 0
        # END Problem 6

    def eat_bee(self, bee):
        # BEGIN Problem 6
        "*** YOUR CODE HERE ***"
        if bee != None:
            bee.reduce_armor(bee.armor)
            self.digesting = 3
        # END Problem 6

    def action(self, colony):
        # BEGIN Problem 6
        "*** YOUR CODE HERE ***"
        if self.digesting > 0:
            self.digesting -= 1
        else:
            self.eat_bee(random_or_none(self.place.bees))
        # END Problem 6

class NinjaAnt(Ant):
    """NinjaAnt does not block the path and damages all bees in its place."""

    name = 'Ninja'
    damage = 1
    blocks_path = False
    # OVERRIDE CLASS ATTRIBUTES HERE
    # BEGIN Problem 7
    implemented = False   # Change to True to view in the GUI
    food_cost = 5
    index = 7
    # END Problem 7

    def action(self, colony):
        # BEGIN Problem 7
        "*** YOUR CODE HERE ***"
        num_bees = len(self.place.bees) - 1
        while num_bees >= 0:
            self.place.bees[num_bees].reduce_armor(1)
            num_bees -= 1
        # END Problem 7

# BEGIN Problem 8
# The WallAnt class
class WallAnt(Ant):
    """WallAnt is an Ant which has a large amount of armor."""

    name = 'Wall'
    "*** YOUR CODE HERE ***"
    # implemented = True
    implemented = False
    food_cost = 4
    index = 8

    def __init__(self):
        "*** YOUR CODE HERE ***"
        Ant.__init__(self, 4)
# END Problem 8

class BodyguardAnt(Ant):
    """BodyguardAnt provides protection to other Ants."""

    name = 'Bodyguard'
    # OVERRIDE CLASS ATTRIBUTES HERE
    # BEGIN Problem 9
    # implemented = True   # Change to True to view in the GUI
    implemented = False
    is_container = True
    food_cost = 4
    index = 9
    # END Problem 9

    def __init__(self, armor=2):
        Ant.__init__(self, armor)
        self.contained_ant = None  # The Ant hidden in this bodyguard

    def can_contain(self, other):
        # BEGIN Problem 9
        "*** YOUR CODE HERE ***"
        if(self.contained_ant is None and other.is_container is False):
            return True
        else:
            return False
        # END Problem 9

    def contain_ant(self, ant):
        # BEGIN Problem 9
        "*** YOUR CODE HERE ***"
        self.contained_ant=ant
        # END Problem 9

    def action(self, colony):
        # BEGIN Problem 9
        "*** YOUR CODE HERE ***"
        if(self.contained_ant is not None):
            self.contained_ant.action(colony)
        # END Problem 9

class TankAnt(BodyguardAnt):
    """TankAnt provides both offensive and defensive capabilities."""

    name = 'Tank'
    damage = 1
    # OVERRIDE CLASS ATTRIBUTES HERE
    # BEGIN Problem 10
    # implemented = True   # Change to True to view in the GUI
    implemented = False
    food_cost = 6
    index = 10
    # END Problem 10

    def action(self, colony):
        # BEGIN Problem 10
        "*** YOUR CODE HERE ***"
        super().action(colony)
        num_bees = len(self.place.bees) - 1
        while num_bees >= 0:
            self.place.bees[num_bees].reduce_armor(damage)
            num_bees -= 1
        # END Problem 10

class Water(Place):
    """Water is a place that can only hold watersafe insects."""

    def add_insect(self, insect):
        """Add an Insect to this place. If the insect is not watersafe, reduce
        its armor to 0."""
        # BEGIN Problem 11
        "*** YOUR CODE HERE ***"
        # END Problem 11

# BEGIN Problem 12
# The ScubaThrower class
# END Problem 12

# BEGIN Problem 13
class QueenAnt(Ant):  # You should change this line
# END Problem 13
    """The Queen of the colony. The game is over if a bee enters her place."""

    name = 'Queen'
    # OVERRIDE CLASS ATTRIBUTES HERE
    # BEGIN Problem 13
    implemented = False   # Change to True to view in the GUI
    # END Problem 13

    def __init__(self, armor=1):
        # BEGIN Problem 13
        "*** YOUR CODE HERE ***"
        # END Problem 13

    def action(self, colony):
        """A queen ant throws a leaf, but also doubles the damage of ants
        in her tunnel.

        Impostor queens do only one thing: reduce their own armor to 0.
        """
        # BEGIN Problem 13
        "*** YOUR CODE HERE ***"
        # END Problem 13

    def reduce_armor(self, amount):
        """Reduce armor by AMOUNT, and if the True QueenAnt has no armor
        remaining, signal the end of the game.
        """
        # BEGIN Problem 13
        "*** YOUR CODE HERE ***"
        # END Problem 13

class AntRemover(Ant):
    """Allows the player to remove ants from the board in the GUI."""

    name = 'Remover'
    implemented = False

    def __init__(self):
        Ant.__init__(self, 0)


##################
# Status Effects #
##################

def make_slow(action, bee):
    """Return a new action method that calls ACTION every other turn.

    action -- An action method of some Bee
    """
    # BEGIN Problem EC
    "*** YOUR CODE HERE ***"
    # END Problem EC

def make_scare(action, bee):
    """Return a new action method that makes the bee go backwards.

    action -- An action method of some Bee
    """
    # BEGIN Problem EC
    "*** YOUR CODE HERE ***"
    # END Problem EC

def apply_effect(effect, bee, duration):
    """Apply a status effect to a BEE that lasts for DURATION turns."""
    # BEGIN Problem EC
    "*** YOUR CODE HERE ***"
    # END Problem EC


class SlowThrower(ThrowerAnt):
    """ThrowerAnt that causes Slow on Bees."""

    name = 'Slow'
    # BEGIN Problem EC
    implemented = False   # Change to True to view in the GUI
    # END Problem EC

    def throw_at(self, target):
        if target:
            apply_effect(make_slow, target, 3)


class ScaryThrower(ThrowerAnt):
    """ThrowerAnt that intimidates Bees, making them back away instead of advancing."""

    name = 'Scary'
    # BEGIN Problem EC
    implemented = False   # Change to True to view in the GUI
    # END Problem EC

    def throw_at(self, target):
        # BEGIN Problem EC
        "*** YOUR CODE HERE ***"
        # END Problem EC

class LaserAnt(ThrowerAnt):
    # This class is optional. Only one test is provided for this class.

    name = 'Laser'
    # OVERRIDE CLASS ATTRIBUTES HERE
    # BEGIN Problem OPTIONAL
    implemented = False   # Change to True to view in the GUI
    # END Problem OPTIONAL

    def __init__(self, armor=1):
        ThrowerAnt.__init__(self, armor)
        self.insects_shot = 0

    def insects_in_front(self, beehive):
        # BEGIN Problem OPTIONAL
        return {}
        # END Problem OPTIONAL

    def calculate_damage(self, distance):
        # BEGIN Problem OPTIONAL
        return 0
        # END Problem OPTIONAL

    def action(self, colony):
        insects_and_distances = self.insects_in_front(colony.beehive)
        for insect, distance in insects_and_distances.items():
            damage = self.calculate_damage(distance)
            insect.reduce_armor(damage)
            if damage:
                self.insects_shot += 1


##################
# Bees Extension #
##################

class Wasp(Bee):
    """Class of Bee that has higher damage."""
    name = 'Wasp'
    damage = 2

class Hornet(Bee):
    """Class of bee that is capable of taking two actions per turn, although
    its overall damage output is lower. Immune to status effects.
    """
    name = 'Hornet'
    damage = 0.25

    def action(self, colony):
        for i in range(2):
            if self.armor > 0:
                super().action(colony)

    def __setattr__(self, name, value):
        if name != 'action':
            object.__setattr__(self, name, value)

class NinjaBee(Bee):
    """A Bee that cannot be blocked. Is capable of moving past all defenses to
    assassinate the Queen.
    """
    name = 'NinjaBee'

    def blocked(self):
        return False

class Boss(Wasp, Hornet):
    """The leader of the bees. Combines the high damage of the Wasp along with
    status effect immunity of Hornets. Damage to the boss is capped up to 8
    damage by a single attack.
    """
    name = 'Boss'
    damage_cap = 8
    action = Wasp.action

    def reduce_armor(self, amount):
        super().reduce_armor(self.damage_modifier(amount))

    def damage_modifier(self, amount):
        return amount * self.damage_cap/(self.damage_cap + amount)

class Hive(Place):
    """The Place from which the Bees launch their assault.

    assault_plan -- An AssaultPlan; when & where bees enter the colony.
    """

    def __init__(self, assault_plan):
        self.name = 'Hive'
        self.assault_plan = assault_plan
        self.bees = []
        for bee in assault_plan.all_bees:
            self.add_insect(bee)
        # The following attributes are always None for a Hive
        self.entrance = None
        self.ant = None
        self.exit = None

    def strategy(self, colony):
        exits = [p for p in colony.places.values() if p.entrance is self]
        for bee in self.assault_plan.get(colony.time, []):
            bee.move_to(random.choice(exits))
            colony.active_bees.append(bee)


class AntColony(object):
    """An ant collective that manages global game state and simulates time.

    Attributes:
    time -- elapsed time
    food -- the colony's available food total
    queen -- the place where the queen resides
    places -- A list of all places in the colony (including a Hive)
    bee_entrances -- A list of places that bees can enter
    """

    def __init__(self, strategy, beehive, ant_types, create_places, dimensions, food=2, random = False):
        """Create an AntColony for simulating a game.

        Arguments:
        strategy -- a function to deploy ants to places
        beehive -- a Hive full of bees
        ant_types -- a list of ant constructors
        create_places -- a function that creates the set of places
        dimensions -- a pair containing the dimensions of the game layout
        """
        self.time = 0
        self.food = food
        self.strategy = strategy
        self.beehive = beehive
        self.ant_types = OrderedDict((a.name, a) for a in ant_types)
        self.dimensions = dimensions
        self.active_bees = []
        self.configure(beehive, create_places)
        # Store the location of ants and bees
        self.antsPlace = np.zeros([dimensions[0], dimensions[1]])
        self.beesPlace = np.zeros([dimensions[0], dimensions[1]])
        self.ant_types_list = self.get_ant_types()
        self.reward = 0
        self.WinReward = 10

        if(random):
            self.ants_random_cost = OrderedDict((a.name, np.random.normal(0, 0.4)) for a in ant_types)
            self.food = self.food + np.random.normal(0.5, 0.5)
        else:
            self.ants_random_cost = OrderedDict((a.name, 0) for a in ant_types)

    def get_ant_types(self):
        ant_types = []
        for name, ant_type in self.ant_types.items():
            ant_types.append({"name": name, "cost": ant_type.food_cost})
        #Sort by cost
        ant_types.sort(key=lambda item: item["cost"])
        return ant_types

    def updateBeesPlace(self):
        for bee in self.active_bees[:]:
            self.beesPlace[bee.place.location[0]][bee.place.location[1]] += 1

    def updateAntsPlace(self):
        self.antsPlace = np.zeros([self.dimensions[0], self.dimensions[1]])
        for ant in self.ants[:]:
            if ant.armor > 0:
                self.antsPlace[ant.place.location[0]][ant.place.location[1]] = ant.index

    def stateReward(self):
        reward = 0
        a = 0.01
        b = 1
        weights = np.array([np.arange(self.dimensions[1], 0, -1) for _ in range(self.dimensions[0])])
        beeReward = np.sum(-weights*self.beesPlace)
        gamma = 1.5
        antReward = np.sum([ant.food_cost * gamma for ant in self.ants])
        reward = (a * beeReward + b * antReward + self.food)*0.01
        self.reward = reward
        return reward
        

    def configure(self, beehive, create_places):
        """Configure the places in the colony."""
        self.base = QueenPlace('AntQueen')
        self.places = OrderedDict()
        self.bee_entrances = []
        def register_place(place, is_bee_entrance):
            self.places[place.name] = place
            if is_bee_entrance:
                place.entrance = beehive
                self.bee_entrances.append(place)
        register_place(self.beehive, False)
        create_places(self.base, register_place, self.dimensions[0], self.dimensions[1])

    def simulate(self):
        """Simulate an attack on the ant colony (i.e., play the game)."""
        num_bees = len(self.bees)
        try:
            while True:
                self.updateBeesPlace()
                self.updateAntsPlace()
                self.strategy(self)                 # Ants deploy
                self.beehive.strategy(self)            # Bees invade
                for ant in self.ants:               # Ants take actions
                    if ant.armor > 0:
                        ant.action(self)
                for bee in self.active_bees[:]:     # Bees take actions
                    if bee.armor > 0:
                        bee.action(self)
                    if bee.armor <= 0:
                        num_bees -= 1
                        self.active_bees.remove(bee)
                if num_bees == 0:
                    raise AntsWinException()
                self.stateReward()
                self.time += 1
        except AntsWinException:
            # print('All bees are vanquished. You win!')
            return True
        except BeesWinException:
            # print('The ant queen has perished. Please try again.')
            return False

    def simulateOnce(self, action, count, isprint):
        """Simulate an attack on the ant colony (i.e., play the game)."""
        num_bees = len(self.bees)
        try:
            self.strategy(self, action)                 # Ants deploy
            self.beehive.strategy(self)            # Bees invade
            for ant in self.ants:               # Ants take actions
                if ant.armor > 0:
                    ant.action(self)
            for bee in self.active_bees[:]:     # Bees take actions
                if bee.armor > 0:
                    bee.action(self)
                if bee.armor <= 0:
                    num_bees -= 1
                    self.active_bees.remove(bee)
            if num_bees == 0:
                raise AntsWinException()
            self.updateBeesPlace()
            self.updateAntsPlace()
            next_observation = [self.beesPlace, self.antsPlace]
            self.stateReward()
            self.time += 1
            if(isprint):
                print('Ant: ')
                print(self.antsPlace)
                print('Bees: ')
                print(self.beesPlace)
                print('Food: ')
                print(self.food)
            return next_observation, self.reward, False
        except AntsWinException:
            if(isprint):
                print('All bees are vanquished. You win!')
            next_observation = [self.beesPlace, self.antsPlace]
            count.winGame()
            return next_observation, self.WinReward, True
        except BeesWinException:
            if(isprint):
                print('The ant queen has perished. Please try again.')
            next_observation = [self.beesPlace, self.antsPlace]
            count.loseGame()
            return next_observation, 0 - self.WinReward, True

    def deploy_ant(self, place_name, ant_type_name):
        """Place an ant if enough food is available.

        This method is called by the current strategy to deploy ants.
        """
        constructor = self.ant_types[ant_type_name]
        extra_random_cost = self.ants_random_cost[ant_type_name]
        if self.food < (constructor.food_cost + extra_random_cost):
            print('Not enough food remains to place ' + ant_type_name)
        else:
            ant = constructor()
            self.places[place_name].add_insect(ant)
            self.food -= (constructor.food_cost + extra_random_cost)
            return ant

    def remove_ant(self, place_name):
        """Remove an Ant from the Colony."""
        place = self.places[place_name]
        if place.ant is not None:
            place.remove_insect(place.ant)

    @property
    def ants(self):
        return [p.ant for p in self.places.values() if p.ant is not None]

    @property
    def bees(self):
        return [b for p in self.places.values() for b in p.bees]

    @property
    def insects(self):
        return self.ants + self.bees

    def __str__(self):
        status = ' (Food: {0}, Time: {1})'.format(self.food, self.time)
        return str([str(i) for i in self.ants + self.bees]) + status

class QueenPlace(Place):
    """QueenPlace at the end of the tunnel, where the queen resides."""

    def add_insect(self, insect):
        """Add an Insect to this Place.

        Can't actually add Ants to a QueenPlace. However, if a Bee attempts to
        enter the QueenPlace, a BeesWinException is raised, signaling the end
        of a game.
        """
        assert not insect.is_ant, 'Cannot add {0} to QueenPlace'
        raise BeesWinException()

def ants_win():
    """Signal that Ants win."""
    raise AntsWinException()

def bees_win():
    """Signal that Bees win."""
    raise BeesWinException()

def ant_types():
    """Return a list of all implemented Ant classes."""
    all_ant_types = []
    new_types = [Ant]
    while new_types:
        new_types = [t for c in new_types for t in c.__subclasses__()]
        all_ant_types.extend(new_types)
    return [t for t in all_ant_types if t.implemented]

class GameOverException(Exception):
    """Base game over Exception."""
    pass

class AntsWinException(GameOverException):
    """Exception to signal that the ants win."""
    pass

class BeesWinException(GameOverException):
    """Exception to signal that the bees win."""
    pass

def interactive_strategy(colony):
    """A strategy that starts an interactive session and lets the user make
    changes to the colony.

    For example, one might deploy a ThrowerAnt to the first tunnel by invoking
    colony.deploy_ant('tunnel_0_0', 'Thrower')
    """
    print('colony: ' + str(colony))
    msg = '<Control>-D (<Control>-Z <Enter> on Windows) completes a turn.\n'
    interact(msg)

def start_with_strategy(args, strategy, random = False):
    """Reads command-line arguments and starts a game with those options."""
    import argparse
    parser = argparse.ArgumentParser(description="Play Ants vs. SomeBees")
    parser.add_argument('-d', type=str, metavar='DIFFICULTY',
                        help='sets difficulty of game (test/easy/medium/hard/extra-hard)')
    parser.add_argument('-w', '--water', action='store_true',
                        help='loads a full layout with water')
    parser.add_argument('--food', type=int,
                        help='number of food to start with when testing', default=2)
    args = parser.parse_args()

    assault_plan = make_normal_assault_plan()
    layout = dry_layout
    tunnel_length = 10
    num_tunnels = 3
    num_tunnels = 1
    food = args.food

    if args.water:
        layout = wet_layout
    if args.d in ['t', 'test']:
        assault_plan = make_test_assault_plan()
        num_tunnels = 1
    elif args.d in ['e', 'easy']:
        assault_plan = make_easy_assault_plan()
        num_tunnels = 2
    elif args.d in ['n', 'normal']:
        assault_plan = make_normal_assault_plan()
        num_tunnels = 3
    elif args.d in ['h', 'hard']:
        assault_plan = make_hard_assault_plan()
        num_tunnels = 4
    elif args.d in ['i', 'extra-hard']:
        assault_plan = make_extra_hard_assault_plan()
        num_tunnels = 4

    beehive = Hive(assault_plan)
    dimensions = (num_tunnels, tunnel_length)
    return AntColony(strategy, beehive, ant_types(), layout, dimensions, food, random)


###########
# Layouts #
###########

def wet_layout(queen, register_place, tunnels=3, length=9, moat_frequency=3):
    """Register a mix of wet and and dry places."""
    for tunnel in range(tunnels):
        exit = queen
        for step in range(length):
            if moat_frequency != 0 and (step + 1) % moat_frequency == 0:
                exit = Water('water_{0}_{1}'.format(tunnel, step), exit)
            else:
                exit = Place('tunnel_{0}_{1}'.format(tunnel, step), exit)
            register_place(exit, step == length - 1)

def dry_layout(queen, register_place, tunnels=3, length=9):
    """Register dry tunnels."""
    wet_layout(queen, register_place, tunnels, length, 0)


#################
# Assault Plans #
#################

class AssaultPlan(dict):
    """The Bees' plan of attack for the Colony.  Attacks come in timed waves.

    An AssaultPlan is a dictionary from times (int) to waves (list of Bees).

    >>> AssaultPlan().add_wave(4, 2)
    {4: [Bee(3, None), Bee(3, None)]}
    """

    def add_wave(self, bee_type, bee_armor, time, count):
        """Add a wave at time with count Bees that have the specified armor."""
        bees = [bee_type(bee_armor) for _ in range(count)]
        self.setdefault(time, []).extend(bees)
        return self

    @property
    def all_bees(self):
        """Place all Bees in the beehive and return the list of Bees."""
        return [bee for wave in self.values() for bee in wave]

def make_test_assault_plan():
    return AssaultPlan().add_wave(Bee, 3, 2, 1).add_wave(Bee, 3, 3, 1)

def make_easy_assault_plan():
    plan = AssaultPlan()
    for time in range(3, 16, 2):
        plan.add_wave(Bee, 3, time, 1)
    plan.add_wave(Wasp, 3, 4, 1)
    plan.add_wave(NinjaBee, 3, 8, 1)
    plan.add_wave(Hornet, 3, 12, 1)
    plan.add_wave(Boss, 15, 16, 1)
    return plan

def make_normal_assault_plan():
    plan = AssaultPlan()
    for time in range(3, 16, 2):
        plan.add_wave(Bee, 3, time, 2)
    plan.add_wave(Wasp, 3, 4, 1)
    plan.add_wave(NinjaBee, 3, 8, 1)
    plan.add_wave(Hornet, 3, 12, 1)
    plan.add_wave(Wasp, 3, 16, 1)

    #Boss Stage
    for time in range(21, 30, 2):
        plan.add_wave(Bee, 3, time, 2)
    plan.add_wave(Wasp, 3, 22, 2)
    plan.add_wave(Hornet, 3, 24, 2)
    plan.add_wave(NinjaBee, 3, 26, 2)
    plan.add_wave(Hornet, 3, 28, 2)
    plan.add_wave(Boss, 20, 30, 1)
    return plan

def make_hard_assault_plan():
    plan = AssaultPlan()
    for time in range(3, 16, 2):
        plan.add_wave(Bee, 4, time, 2)
    plan.add_wave(Hornet, 4, 4, 2)
    plan.add_wave(Wasp, 4, 8, 2)
    plan.add_wave(NinjaBee, 4, 12, 2)
    plan.add_wave(Wasp, 4, 16, 2)

    #Boss Stage
    for time in range(21, 30, 2):
        plan.add_wave(Bee, 4, time, 3)
    plan.add_wave(Wasp, 4, 22, 2)
    plan.add_wave(Hornet, 4, 24, 2)
    plan.add_wave(NinjaBee, 4, 26, 2)
    plan.add_wave(Hornet, 4, 28, 2)
    plan.add_wave(Boss, 30, 30, 1)
    return plan

def make_extra_hard_assault_plan():
    plan = AssaultPlan()
    plan.add_wave(Hornet, 5, 2, 2)
    for time in range(3, 16, 2):
        plan.add_wave(Bee, 5, time, 2)
    plan.add_wave(Hornet, 5, 4, 2)
    plan.add_wave(Wasp, 5, 8, 2)
    plan.add_wave(NinjaBee, 5, 12, 2)
    plan.add_wave(Wasp, 5, 16, 2)

    #Boss Stage
    for time in range(21, 30, 2):
        plan.add_wave(Bee, 5, time, 3)
    plan.add_wave(Wasp, 5, 22, 2)
    plan.add_wave(Hornet, 5, 24, 2)
    plan.add_wave(NinjaBee, 5, 26, 2)
    plan.add_wave(Hornet, 5, 28, 2)
    plan.add_wave(Boss, 30, 30, 2)
    return plan

# random strategy
from randomStrategy import *
from SimpleDQN import *
from utils import *
import matplotlib.pyplot as plt
import twoStepDQN2 as T2
import twoStepDQN2_MAML as T2MAML

class CountNumber():
    def  __init__(self):
        self.win = 0
        self.lose = 0
    
    def reset(self):
        self.win = 0
        self.lose = 0

    def print(self):
        print('win = {}, lose = {}'.format(self.win, self.lose))

    def winGame(self):
        self.win += 1

    def loseGame(self):
        self.lose += 1

@main
def run(*args):
    Insect.reduce_armor = class_method_wrapper(Insect.reduce_armor,
            pre=print_expired_insects)
    # start_with_strategy(args, interactive_strategy)
    myCount = CountNumber()
    antcolony = start_with_strategy(args, actionToStrategy)
    '''
    This is Simple DQN
    '''
    # agent = DQNAgent(antcolony)
    # episodes = 600
    # episode_rewards = []
    # for episode in range(episodes):
    #     episode_reward, loss = play_qlearning(start_with_strategy(args, actionToStrategy), agent, myCount, False, train=True, update=(episode%5==0))
    #     episode_rewards.append(episode_reward)
    #     print(episode, ':', episode_reward)
    # # np.save('variables/ants_SimpleDDQN_training_episode_reward.npy', episode_rewards)
    # plt.figure()
    # plt.plot(episode_rewards)
    # plt.show()
    # myCount.print()
    # print('----------------------Test------------------------')
    # myCount.reset()

    # agent.epsilon = 0. 
    # episode_rewards = [play_qlearning(start_with_strategy(args, actionToStrategy), agent, myCount, True) for _ in range(1)]
    '''
    This is Two-Step DQN
    '''
    # print('This is Two-Step DQN')
    # agent = T2.TwoStepDQNAgent(antcolony, epsilon=0.4)
    # agent.trainState = 2
    # episodes = 600
    # episode_rewards_step1 = []
    # losses=[]
    # for episode in range(episodes):
    #     episode_reward, loss = play_qlearning(start_with_strategy(args, actionToStrategy), agent, myCount, False, train=True, update=(episode%5==0))
    #     print('Episode: ', episode, '  Reward: ', episode_reward, ' Loss: ', loss)
    #     episode_rewards_step1.append(episode_reward)
    #     losses.append(loss)
    # # np.save('variables/ants_DDQN_training_episode_reward_together_normal.npy', episode_rewards_step1)
    
    '''
    This is Two-Step DQN (Train separately)
    '''
    # print('This is Two-Step DQN (Train separately)')
    # agent = T2.TwoStepDQNAgent(antcolony, epsilon=0.4)
    # episodes = 300
    # episode_rewards_step1 = []
    # losses=[]
    # for episode in range(episodes):
    #     episode_reward, loss = play_qlearning(start_with_strategy(args, actionToStrategy), agent, myCount, False, train=True, update=(episode%5==0))
    #     # if(episode%30==0):
    #         # agent.evaluate_net_type.save_weights('Weights_ants/maml_ants'+str(episode)+'.h5')
    #     print('Step 1 Episode: ', episode, '  Reward: ', episode_reward, ' Loss: ', loss)
    #     episode_rewards_step1.append(episode_reward)
    #     losses.append(loss)
    # # np.save('variables/ants_DDQN_training_episode_reward_step_1_normal.npy', episode_rewards_step1)
    # # agent.evaluate_net_type.save_weights('maml_ants_normal_1_'+str(episode)+'.h5')
    # myCount.print()
    # print('----------------------Test------------------------')
    # myCount.reset()
    # # agent.epsilon = 0. 
    # # episode_rewards = [play_qlearning(start_with_strategy(args, actionToStrategy), agent, myCount) for _ in tqdm(range(100))]
    # agent.epsilon = 0. 
    # episode_rewards = [play_qlearning(start_with_strategy(args, actionToStrategy), agent, myCount, True) for _ in range(1)]
    # agent.epsilon=0.4



    # episode_rewards_step2 = []
    # episodes = 500
    # agent.trainState = 1
    # for episode in range(episodes):
    #     episode_reward, loss = play_qlearning(start_with_strategy(args, actionToStrategy), agent, myCount, False, train=True, update=(episode%5==0))
    #     # if(episode%30==0):
    #     #     for i, layer in enumerate(agent.evaluate_net_position):
    #     #         layer.save_weights('Weights_ants_step2/maml_ants_step2'+str(episode)+'_'+str(i)+'.h5')
    #     print('Step 2 Episode: ', episode, '  Reward: ', episode_reward, ' Loss: ', loss)
    #     episode_rewards_step2.append(episode_reward)
    #     # losses.append(loss)
    # # np.save('variables/ants_DDQN_training_episode_reward_step_2_normal.npy', episode_rewards_step2)

    # myCount.print()
    # print('----------------------Test------------------------')
    # myCount.reset()
    # agent.epsilon = 0. 
    # episode_rewards = [play_qlearning(start_with_strategy(args, actionToStrategy), agent, myCount, True) for _ in range(1)]
    # agent.epsilon=0.4

    # episode_rewards_step3 = []
    # episodes = 300
    # agent.trainState = 2
    # for episode in range(episodes):
    #     episode_reward, loss = play_qlearning(start_with_strategy(args, actionToStrategy), agent, myCount, False, train=True, update=(episode%5==0))
    #     # if(episode%30==0):
    #     #     for i, layer in enumerate(agent.evaluate_net_position):
    #     #         layer.save_weights('Weights_ants_step2/maml_ants_step2'+str(episode)+'_'+str(i)+'.h5')
    #     print('Step 3 Episode: ', episode, '  Reward: ', episode_reward, ' Loss: ', loss)
    #     episode_rewards_step3.append(episode_reward)
    #     # losses.append(loss)
    # # np.save('variables/ants_DDQN_training_episode_reward_step_3_normal.npy', episode_rewards_step3)
    # myCount.print()
    # print('----------------------Test------------------------')
    # myCount.reset()
    # agent.epsilon = 0. 
    # episode_rewards = [play_qlearning(start_with_strategy(args, actionToStrategy), agent, myCount, True) for _ in range(1)]

    '''
    This is Two-Step DQN MAML Learning
    '''
    # tf.keras.backend.set_floatx('float64')
    # agent = T2MAML.TwoStepDQNAgent(antcolony, myCount, args, actionToStrategy, start_with_strategy, epsilon=0.4)
    # # agent.trainState = 2
    # episodes = 600
    # last_episode=[]
    # for episode in range(episodes):
    #    agent.trainOnce()
    #    print(episode, ' Last episode: ', agent.episodes_reward[-1])
    #    last_episode.append(agent.episodes_reward[-1])
    # #    if(episode%30==0):
    # #         agent.evaluate_net_type.save_weights('MAML_weights_ants/maml_ants'+str(episode)+'.h5')
    # myCount.print()
    # agent.validation_net_position = agent.evaluate_net_position
    # agent.validation_net_type = agent.evaluate_net_type
    # # np.save('variables/ants_MAML_training_episode_reward.npy', agent.episodes_reward)
    # # np.save('variables/ants_MAML_training_last_episode_reward.npy', last_episode)
    # print('----------------------Test------------------------')
    # myCount.reset()
    # agent.validation(10, start_with_strategy(args, actionToStrategy, True))



    # episode_rewards_step2 = []
    # episodes = 500
    # agent.trainState = 1
    # for episode in range(episodes):
    #     episode_reward, loss = play_qlearning(start_with_strategy(args, actionToStrategy), agent, myCount, False, train=True, update=(episode%5==0))
    #     print('Episode: ', episode, '  Reward: ', episode_reward, ' Loss: ', loss)
    #     episode_rewards_step2.append(episode_reward)
    # plt.figure()
    # plt.plot(agent.episodes_reward)
    # plt.show()

    # # agent.evaluate_net_type.save_weights('ants_net_type.h5')

    # plt.figure()
    # plt.plot(episode_rewards_step2)
    # plt.show()
    '''
    This is comparision for MAML and fine-tuning.
    '''
    # tf.keras.backend.set_floatx('float64')
    # agent_MAML = T2MAML.TwoStepDQNAgent(antcolony, myCount, args, actionToStrategy, start_with_strategy, epsilon=0.4)
    # agent_finetuning = T2MAML.TwoStepDQNAgent(antcolony, myCount, args, actionToStrategy, start_with_strategy, epsilon=0.4)
    # observation = [antcolony.beesPlace, antcolony.antsPlace]
    # observation = np.array(observation).transpose(1,2,0)
    # agent_MAML.validation_net_type.forward(observation[np.newaxis])
    # agent_finetuning.validation_net_type.forward(observation[np.newaxis])
    # episode = 2
    # averageRewardMAML=[]
    # averageRewardFineTuning=[]
    # for i in range(20):
    #     reward_MAML = []
    #     reward_finetuning = []
    #     for j in range(10):
    #         agent_MAML.validation_net_type.load_weights('MAML_weights_ants/maml_ants'+str(i*30)+'.h5')
    #         agent_finetuning.validation_net_type.load_weights('Weights_ants/maml_ants'+str(i*30)+'.h5')
    #         colony_MAML = start_with_strategy(args, actionToStrategy, True)
    #         colony_finetuning = copy.deepcopy(colony_MAML)
    #         reward_MAML.append(agent_MAML.validation(episode, colony_MAML))
    #         reward_finetuning.append(agent_finetuning.validation(episode, colony_finetuning))
    #     averageRewardMAML.append(np.mean(reward_MAML))
    #     averageRewardFineTuning.append(np.mean(reward_finetuning))
    #     print(i, 'MAML:', averageRewardMAML[-1],'fine tuning:',averageRewardFineTuning[-1])
    # plt.figure()
    # plt.plot(averageRewardMAML)
    # plt.plot(averageRewardFineTuning)
    # plt.show()
    # # np.save('variables/ants_comparision_MAML_finetuning.npy', [averageRewardMAML, averageRewardFineTuning])