from rlmodels.dqn import Transition
import torch
import torch.nn as nn
def optimize_model(memory,device,policy_model,traget_model,optimizer,gamma = 0.8,batch_size=32):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    for item in batch:
      if torch.is_tensor(item):
        item = item.to(device)

    mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    
    next_states = torch.stack([torch.tensor(s) for s in batch.next_state
                                                if s is not None])
    
    state_batch = torch.stack(tuple([torch.tensor(s) for s in list(batch.state)]))
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_vals = policy_model(state_batch).gather(1, action_batch)
    next_state_vals = torch.zeros(batch_size, device=device)
    next_state_vals[mask] = traget_model(next_states).max(1)[0].detach()
    # Compute the expected Q values
    state_action_expected_vals = (next_state_vals * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_vals, state_action_expected_vals.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()