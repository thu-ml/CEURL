import os
import pandas as pd
import numpy as np
import argparse


# Hyperparameters
def get_parser():
    parser = argparse.ArgumentParser(description='PlaNet or Dreamer')
    parser.add_argument('--domains', type=str, default='walker_mass2')
    return parser.parse_args()


def list_subdirectories(folder_path):
    subdirectories = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    return subdirectories


args = get_parser()
base_dir = './exp_local'
domains = args.domains.split('~')
tasks = {
    'walker_mass': ['finetune_walker_stand_mass', 'finetune_walker_walk_mass',
                     'finetune_walker_run_mass', 'finetune_walker_flip_mass'],
    'quadruped_mass': ['finetune_quadruped_stand_mass', 'finetune_quadruped_walk_mass',
                       'finetune_quadruped_run_mass', 'finetune_quadruped_jump_mass'],
    'quadruped_damping': ['finetune_quadruped_stand_damping', 'finetune_quadruped_walk_damping',
                           'finetune_quadruped_run_damping', 'finetune_quadruped_jump_damping'],
}
algs = {
    'walker_mass': [
        'peac',
    ],
    'quadruped_mass': [
        'peac',
    ],
    'quadruped_damping': [
        'peac',
    ],
}
snapshot_ts = {
    'walker_mass': ['2000000'],
    'quadruped_mass': ['2000000'],
    'quadruped_damping': ['2000000'],
}

for domain in domains:
    for alg in algs[domain]:
        for sns in snapshot_ts[domain]:
            print('*'*10, 'current domain:', domain, 'current algo:', alg, 'current snapshot_ts:',
                  sns, '*'*10)
            best_train_results = []
            best_eval_results = []
            last_train_results = []
            last_eval_results = []
            for task in tasks[domain]:
                print('current task:', task)
                folder_path = os.path.join(base_dir, domain, task, alg, sns)
                subdirectories = list_subdirectories(folder_path)
                subdirectories = [a + '/eval.csv' for a in subdirectories]
                # print(subdirectories)
                train_results = []
                eval_results = []
                for sub_dict in subdirectories:
                    # print(sub_dict)
                    # if '_1/eval.csv' not in sub_dict:
                    #     continue
                    df = pd.read_csv(sub_dict)
                    episodes = df['episode'].tolist()
                    train_result = df['episode_train_reward'].tolist()
                    eval_result = df['episode_eval_reward'].tolist()
                    if len(eval_result) != 11:
                        print(sub_dict, len(eval_result))
                        continue

                    train_results.append(train_result)
                    eval_results.append(eval_result)

                    # print(len(eval_result))
                train_results = np.mean(train_results, axis=0)
                eval_results = np.mean(eval_results, axis=0)
                index = np.argmax(train_results)
                print('train last:', train_results[-1], 'best index:', index,
                      'best:', train_results[index])
                print('eval last:', eval_results[-1], 'best index:', index,
                      'best:', eval_results[index])
                last_train_results.append(train_results[-1])
                best_train_results.append(train_results[index])
                last_eval_results.append(eval_results[-1])
                best_eval_results.append(eval_results[index])
            print('average of all tasks')
            print('train last:', np.mean(last_train_results),
                  'best:', np.mean(best_train_results))
            print('eval last:', np.mean(last_eval_results),
                  'best:', np.mean(best_eval_results))
