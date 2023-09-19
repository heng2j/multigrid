from multigrid.agents_pool.YourName_policies import YourNamePolicy


SubmissionPolicies = {
    "YourNamePolicy": YourNamePolicy,
}




# This should be a mapping of roles in a substrate to a list of policy_ids
# If all the entire population works for any role, this dictionary can be left unchanged
SubmissionRoles = {
    "YourNamePolicy":  
        {
            'player_who_likes_red': [f'agent_{i}' for i in range(16)],
            'player_who_likes_green': [f'agent_{i}' for i in range(16)],
        },
}