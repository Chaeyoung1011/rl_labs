import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
from gyms.simple_maze_grid import SimpleMazeGrid
import pygame


def state_to_index(state, n):
    player_pos = np.argwhere(state == 1)
    if player_pos.size == 0:
        return None
    row, col = player_pos[0]
    return row * n + col


def policy_evaluation(env, policy, gamma=0.99, theta=1e-6, render_each_sweep=True):
    V = np.zeros(env.n * env.n)
    sweep = 0

    while True:
        delta = 0
        for player_pos in env.get_all_states():
            env.set_player_pos(player_pos)
            state = env._get_state()
            state_idx = state_to_index(state, env.n)
            if state_idx is None:
                continue

            v_old = V[state_idx]
            v_new = 0.0

            for action, action_prob in enumerate(policy[state_idx]):
                if action_prob <= 0:
                    continue

                _next_state, reward, terminated = env.simulate_action(list(player_pos), action)
                next_state_idx = state_to_index(_next_state, env.n)

                # 종료면 V(s')는 0으로 처리
                next_v = 0.0 if (terminated or next_state_idx is None) else V[next_state_idx]
                v_new += action_prob * (reward + gamma * next_v)

            V[state_idx] = v_new
            delta = max(delta, np.abs(v_old - v_new))

        sweep += 1
        if render_each_sweep and env.render_option:
            env.render_v_values(V, policy, f"Eval {sweep}")
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return V

        if delta < theta:
            break

    return V


def policy_improvement(env, V, gamma=0.99):
    policy = np.zeros((env.n * env.n, env.action_space.n))
    for player_pos in env.get_all_states():
        env.set_player_pos(player_pos)
        state = env._get_state()
        state_idx = state_to_index(state, env.n)
        if state_idx is None:
            continue

        action_values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            _next_state, reward, terminated = env.simulate_action(list(player_pos), action)
            next_state_idx = state_to_index(_next_state, env.n)
            next_v = 0.0 if (terminated or next_state_idx is None) else V[next_state_idx]
            action_values[action] = reward + gamma * next_v

        best_action = int(np.argmax(action_values))
        policy[state_idx, best_action] = 1.0

    return policy


def policy_iteration(env, gamma=0.99, theta=1e-6, render_progress=True):
    policy = np.ones((env.n * env.n, env.action_space.n)) / env.action_space.n
    iteration = 0

    while True:
        print(f"Iteration: {iteration}")
        V = policy_evaluation(env, policy, gamma=gamma, theta=theta,
                              render_each_sweep=render_progress)

        new_policy = policy_improvement(env, V, gamma=gamma)

        if render_progress and env.render_option:
            env.render_v_values(V, new_policy, iteration + 1)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    break

        if np.all(policy == new_policy):
            break

        policy = new_policy
        iteration += 1

    return policy, V, iteration


def main():
    # 요청하신 (1) 케이스로 기본값 설정
    n = 20
    k = 19
    m = 100
    random_seed = 2
    gamma = 0.9
    theta = 1e-6

    env = SimpleMazeGrid(n=n, k=k, m=m, render_option=True, random_seed=random_seed)
    env.render()

    policy, V, iteration = policy_iteration(env, gamma=gamma, theta=theta, render_progress=True)

    print(f"\n정책 반복(Policy Iteration)이 {iteration}번 만에 최적 정책을 찾았습니다.")

    plt.imshow(V.reshape((n, n)), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Value Function')
    plt.savefig('policy_iteration_simple_maze_grid.png')
    plt.show()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    env.close()


if __name__ == "__main__":
    main()
