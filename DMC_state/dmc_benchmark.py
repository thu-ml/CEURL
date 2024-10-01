DOMAINS = [
    'walker',
    'quadruped',
    'jaco',
]

WALKER_TASKS = [
    'walker_stand',
    'walker_walk',
    'walker_run',
    'walker_flip',
]

QUADRUPED_TASKS = [
    'quadruped_walk',
    'quadruped_run',
    'quadruped_stand',
    'quadruped_jump',
]

JACO_TASKS = [
    'jaco_reach_top_left',
    'jaco_reach_top_right',
    'jaco_reach_bottom_left',
    'jaco_reach_bottom_right',
]

TASKS = WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS

parameter_1 = ['0.2', '0.6', '1.0', '1.4', '1.8']
parameter_1_eval = ['0.4', '0.8', '1.2', '1.6']

parameter_2 = ['0.4', '0.8', '1.0', '1.4']
parameter_2_eval = ['0.6', '1.2']

PRETRAIN_TASKS = {
    'walker': 'walker_stand',
    'jaco': 'jaco_reach_top_left',
    'quadruped': 'quadruped_walk',
    'walker_mass': ['walker_stand~mass~' + para for para in parameter_1],
    'quadruped_mass': ['quadruped_stand~mass~' + para for para in parameter_2],
    'quadruped_damping': ['quadruped_stand~damping~' + para for para in parameter_1],
}

FINETUNE_TASKS = {
    'walker_stand_mass': ['walker_stand~mass~' + para for para in parameter_1],
    'walker_stand_mass_eval': ['walker_stand~mass~' + para for para in parameter_1_eval],
    'walker_walk_mass': ['walker_walk~mass~' + para for para in parameter_1],
    'walker_walk_mass_eval': ['walker_walk~mass~' + para for para in parameter_1_eval],
    'walker_run_mass': ['walker_run~mass~' + para for para in parameter_1],
    'walker_run_mass_eval': ['walker_run~mass~' + para for para in parameter_1_eval],
    'walker_flip_mass': ['walker_flip~mass~' + para for para in parameter_1],
    'walker_flip_mass_eval': ['walker_flip~mass~' + para for para in parameter_1_eval],

    'quadruped_stand_mass': ['quadruped_stand~mass~' + para for para in parameter_2],
    'quadruped_stand_mass_eval': ['quadruped_stand~mass~' + para for para in parameter_2_eval],
    'quadruped_walk_mass': ['quadruped_walk~mass~' + para for para in parameter_2],
    'quadruped_walk_mass_eval': ['quadruped_walk~mass~' + para for para in parameter_2_eval],
    'quadruped_run_mass': ['quadruped_run~mass~' + para for para in parameter_2],
    'quadruped_run_mass_eval': ['quadruped_run~mass~' + para for para in parameter_2_eval],
    'quadruped_jump_mass': ['quadruped_jump~mass~' + para for para in parameter_2],
    'quadruped_jump_mass_eval': ['quadruped_jump~mass~' + para for para in parameter_2_eval],

    'quadruped_stand_damping': ['quadruped_stand~damping~' + para for para in parameter_1],
    'quadruped_stand_damping_eval': ['quadruped_stand~damping~' + para for para in parameter_1_eval],
    'quadruped_walk_damping': ['quadruped_walk~damping~' + para for para in parameter_1],
    'quadruped_walk_damping_eval': ['quadruped_walk~damping~' + para for para in parameter_1_eval],
    'quadruped_run_damping': ['quadruped_run~damping~' + para for para in parameter_1],
    'quadruped_run_damping_eval': ['quadruped_run~damping~' + para for para in parameter_1_eval],
    'quadruped_jump_damping': ['quadruped_jump~damping~' + para for para in parameter_1],
    'quadruped_jump_damping_eval': ['quadruped_jump~damping~' + para for para in parameter_1_eval],
}
