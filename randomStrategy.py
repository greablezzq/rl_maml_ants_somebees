import numpy as np
import random

def get_ant_types(colony):
        ant_types = []
        for name, ant_type in colony.ant_types.items():
            ant_types.append({"name": name, "cost": ant_type.food_cost})
        #Sort by cost
        ant_types.sort(key=lambda item: item["cost"])
        return ant_types

def randomStrateqy(colony):
    # colony.deploy_ant('tunnel_0_0', 'Thrower')
    num_tunnels = colony.dimensions[0]
    tunnel_length = colony.dimensions[1]
    ants = get_ant_types(colony)
    success = False
    while not success:
        randnum = random.randint(0, num_tunnels - 1)
        randlen = random.randint(0, tunnel_length - 1)
        if(colony.antsPlace[randnum][randlen] == 0):
            placeName = 'tunnel_{0}_{1}'.format(randnum, randlen)
            antIndex = random.randint(0,len(ants)-1)
            while antIndex >= 0:
                ant = ants[antIndex]
                antName = ant["name"]
                if(colony.food >= ant["cost"]):
                    colony.deploy_ant(placeName, antName)
                    success = True
                    antIndex = -1
                else:
                    antIndex = antIndex - 1
            success = True
    return


