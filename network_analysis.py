#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 23:08:39 2020

@author: Neeraj

Description: Demonstrates how to do simple network analysis in Python. 
Reference: Chapter 22 : Network Analysis
"""
from typing import NamedTuple

class User(NamedTuple):
    id: int
    name: str
        
        
users = [User(0, "Hero"), User(1,"Dunn"), User(2, "Sue"), User(3,"Chi"), User(4, "Thor"), User(5, "Clive"),
        User(6, "Hicks"), User(7,"Devin"), User(8,"Kate"), User(9,"Klein")]

friend_pairs = [(0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(5,7),(6,8),(7,8),(8,9)]

from typing import Dict, List

# Type aliases for keeping track if Friendships
Friendships = Dict[int, List[int]]

friendships: Friendships = {user.id: [] for user in users}
    
for i, j in friend_pairs:
    friendships[i].append(j)
    friendships[j].append(i)
    
#friendships

# Breadth first search algorithm for finding all shortest paths

from collections import deque
Path = List[int]

def shortest_paths_from(from_user_id: int,
                        friendships: Friendships) -> Dict[int, List[Path]]:
    # A dictionary from "user_id" to *all* shortest paths to that user
    shortest_paths_to: Dict[int, List[Path]] = {from_user_id: [[]]}

    # A queue of (previous user, next user) that we need to check.
    # Starts out with all pairs (from_user, friend_of_from_user)
    frontier = deque((from_user_id, friend_id)
                     for friend_id in friendships[from_user_id])

    # Keep going until we empty the queue.
    while frontier:
        # Remove the pair that's next in the queue.
        prev_user_id, user_id = frontier.popleft()

        # Because of the way we're adding to the queue,
        # necessarily we already know some shortest paths to prev_user
        paths_to_prev_user = shortest_paths_to[prev_user_id]
        new_paths_to_user = [path + [user_id] for path in paths_to_prev_user]

        # It's possible we already know a shortest path to user_id.
        old_paths_to_user = shortest_paths_to.get(user_id, [])

        # What's the shortest path to here that we've seen so far?
        if old_paths_to_user:
            min_path_length = len(old_paths_to_user[0])
        else:
            min_path_length = float('inf')

        # Only keep paths that aren't too long and are actually new
        new_paths_to_user = [path
                             for path in new_paths_to_user
                             if len(path) <= min_path_length
                             and path not in old_paths_to_user]

        shortest_paths_to[user_id] = old_paths_to_user + new_paths_to_user

        # Add never-seen neighbors to the frontier
        frontier.extend((user_id, friend_id)
                        for friend_id in friendships[user_id]
                        if friend_id not in shortest_paths_to)

    return shortest_paths_to

# For each from_user, for each to_user, a list of shortest paths.
shortest_paths = {user.id: shortest_paths_from(user.id, friendships)
                  for user in users}

betweenness_centrality = {user.id: 0.0 for user in users}
for source in users:
    for target_id, paths in shortest_paths[source.id].items():
        if source.id < target_id: # don't double the count
            num_paths = len(paths) # how many shortest paths
            contrib = 1/num_paths  # Contribution to centrality
            for path in paths:
                for between_id in path:
                    if between_id not in [source.id, target_id]:
                        betweenness_centrality[between_id] += contrib
                        
#betweenness_centrality
