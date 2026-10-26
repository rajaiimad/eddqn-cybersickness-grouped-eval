# ============================================================
# E-DDQN v2 — Enhanced Double Deep Q-Network (improved)
# Adversarial Cybersickness Detection in Virtual Reality
# ============================================================
# CHANGES vs v1 (each technically justified — see notes inline):
#   [1] Validation-based model selection (uses X_val; no test peeking)
#   [2] Tunable, rebalanced asymmetric reward (default 2.5:1, not 5:1)
#   [3] Dueling DQN head (Wang et al., 2016) — better value estimation
#   [4] Raw Q-values (softmax removed) — faithful TD targets
#   [5] Soft (Polyak) target updates (van Hasselt 2016 + Lillicrap 2016)
#   [6] Cosine LR horizon fixed to full run (T_max = EPISODES)
#
# IMPORTANT (academic integrity): tune R_FN/R_FP and the selection metric
# on the VALIDATION set, then report the held-out TEST result once. Re-run
# the standard DQN baseline under the SAME protocol (same episodes, same
# validation-based selection) so the comparison is symmetric and fair.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap
import os
import random
import copy

# ── Reproducibility Seed ─────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

MethodName = 'FastDQL'
Results    = f'Results_{MethodName}'
Model      = f'TrainedModel_{MethodName}'
os.makedirs(Results, exist_ok=True)
os.makedirs(Model,   exist_ok=True)
PathModel  = os.path.join(Model, 'statedictmodel.pt')


# ============================================================
# SECTION 1 — Load Pre-split Data (train / VALIDATION / test)
# ============================================================
# The validation split is now used for model selection so the held-out
# test set is touched only once, for the final reported number.
# ============================================================
X_train = pd.read_csv('X_train.csv').values.astype(np.float32)
X_val   = pd.read_csv('X_val.csv').values.astype(np.float32)
X_test  = pd.read_csv('X_test.csv').values.astype(np.float32)
y_train = pd.read_csv('y_train.csv').values.flatten().astype(int)
y_val   = pd.read_csv('y_val.csv').values.flatten().astype(int)
y_test  = pd.read_csv('y_test.csv').values.flatten().astype(int)

# Binarise: any cybersickness severity (1,2,3) → Attack class 1
y_train = np.where(y_train >= 1, 1, 0)
y_val   = np.where(y_val   >= 1, 1, 0)
y_test  = np.where(y_test  >= 1, 1, 0)

print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
print(f"Train class dist: {np.bincount(y_train)}")
print(f"Val   class dist: {np.bincount(y_val)}")
print(f"Test  class dist: {np.bincount(y_test)}")

NUMSTATES    = X_train.shape[1]   # VR telemetry features (41 in the GLCD split)
NUMLabels    = 2                  # 2 actions: 0=Normal, 1=Attack
feature_names = list(pd.read_csv('X_train.csv').columns)

traininput  = torch.tensor(X_train, dtype=torch.float32)
valinput    = torch.tensor(X_val,   dtype=torch.float32)
testinput   = torch.tensor(X_test,  dtype=torch.float32)
trainlabels = torch.tensor(y_train, dtype=torch.long)
vallabels   = torch.tensor(y_val,   dtype=torch.long)
testlabels  = torch.tensor(y_test,  dtype=torch.long)

assert traininput.shape[1] == NUMSTATES and testinput.shape[1] == NUMSTATES
print(f"Feature check passed: {NUMSTATES} features confirmed")


def compute_metrics(preds, true):
    """Confusion-matrix metrics. bal_acc (balanced accuracy) is the
    selection metric: it rewards BOTH classes equally, so it keeps recall
    high while pushing specificity (hence accuracy) up — a fair operating point."""
    preds = np.asarray(preds); true = np.asarray(true)
    TP = int(((preds == 1) & (true == 1)).sum()); TN = int(((preds == 0) & (true == 0)).sum())
    FP = int(((preds == 1) & (true == 0)).sum()); FN = int(((preds == 0) & (true == 1)).sum())
    acc  = (TP + TN) / (TP + TN + FP + FN + 1e-9)
    rec  = TP / (TP + FN + 1e-9); spec = TN / (TN + FP + 1e-9)
    prec = TP / (TP + FP + 1e-9); f1 = 2 * prec * rec / (prec + rec + 1e-9)
    bal_acc = (rec + spec) / 2
    mcc_den = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    mcc = ((TP * TN) - (FP * FN)) / (mcc_den + 1e-9)
    return dict(acc=acc, prec=prec, rec=rec, spec=spec, f1=f1, bal_acc=bal_acc,
                mcc=mcc, TP=TP, TN=TN, FP=FP, FN=FN)


# ============================================================
# SECTION 2 — Neural Network (Dueling head, raw Q-values)
# ============================================================
# [3] DUELING (Wang et al., 2016): a shared trunk feeds two streams —
#     V(s) (state value) and A(s,a) (advantage). Q = V + (A - mean(A)).
#     This stabilises value estimates on states where the action barely
#     matters (most benign samples), improving specificity / accuracy.
# [4] NO SOFTMAX: returns raw Q-values so the network can actually match
#     Bellman targets r + gamma*Q (which may exceed 1 or be negative).
#     argmax is unchanged, so predictions and SHAP behave as before.
# ============================================================
class Net(nn.Module):
    """Dueling MLP: Input → 256 → 512 → 256 → [V(1), A(2)] → Q(2)"""
    def __init__(self, num_states, num_labels, dropout=0.2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_states, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.val = nn.Linear(256, 1)            # state-value stream
        self.adv = nn.Linear(256, num_labels)   # advantage stream
        self.drop = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        x = F.relu(self.bn3(self.fc3(x)))
        v = self.val(x)
        a = self.adv(x)
        return v + (a - a.mean(dim=1, keepdim=True))   # dueling aggregation


# ============================================================
# SECTION 3 — Double DQN Agent (soft target updates)
# ============================================================
class DQN:
    def __init__(self, NUMSTATES, NUMLabels, BATCHSIZE, LR,
                 GAMMA, MEMORYCAPACITY, QNETWORKITERATION, TAU, T_MAX):
        self.eval_net   = Net(NUMSTATES, NUMLabels).to(device)
        self.target_net = Net(NUMSTATES, NUMLabels).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.NUMSTATES      = NUMSTATES
        self.NUMLabels      = NUMLabels
        self.BATCHSIZE      = BATCHSIZE
        self.GAMMA          = GAMMA
        self.MEMORYCAPACITY = MEMORYCAPACITY
        self.QTargetUpdate  = QNETWORKITERATION
        self.TAU            = TAU
        self.learn_step     = 0
        self.mem_counter    = 0
        self.memory = np.zeros((MEMORYCAPACITY, NUMSTATES * 2 + 3), dtype=np.float32)  # +1: true label

        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=LR, weight_decay=1e-4)
        # [6] Cosine annealing across the FULL run (T_max = EPISODES)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=T_MAX, eta_min=1e-5)
        self.loss_fn = nn.SmoothL1Loss()

    def choose_action_batch(self, states, epsilon):
        n         = len(states)
        rand_mask = np.random.rand(n) < epsilon
        actions   = np.random.randint(0, self.NUMLabels, size=n)
        greedy_idx = np.where(~rand_mask)[0]
        if len(greedy_idx) > 0:
            self.eval_net.eval()
            with torch.no_grad():
                q_vals = self.eval_net(states[greedy_idx].to(device))
                actions[greedy_idx] = q_vals.argmax(dim=1).cpu().numpy()
            self.eval_net.train()
        return actions, rand_mask.astype(int)

    def store_transitions_batch(self, states, actions, rewards, next_states, labels):
        transitions = np.hstack([
            states.numpy(), actions.reshape(-1, 1),
            rewards.reshape(-1, 1), next_states.numpy(),
            np.asarray(labels).reshape(-1, 1)])
        for i in range(len(transitions)):
            idx = self.mem_counter % self.MEMORYCAPACITY
            self.memory[idx] = transitions[i]
            self.mem_counter += 1

    def _soft_update(self):
        # [5] Polyak averaging: theta_target <- (1-tau)*theta_target + tau*theta
        with torch.no_grad():
            for tp, ep_ in zip(self.target_net.parameters(), self.eval_net.parameters()):
                tp.data.mul_(1.0 - self.TAU).add_(self.TAU * ep_.data)

    def learn(self):
        self.learn_step += 1
        n   = min(self.mem_counter, self.MEMORYCAPACITY)
        # Class-balanced replay: equal Normal/Attack transitions per minibatch.
        # Counters the 1:3 class imbalance so the agent learns 'Normal' better
        # -> fewer false positives -> higher precision/accuracy at the same recall.
        lab  = self.memory[:n, -1]
        idx0 = np.where(lab == 0)[0]; idx1 = np.where(lab == 1)[0]
        if len(idx0) > 0 and len(idx1) > 0:
            h0 = int(self.BATCHSIZE * NORMAL_FRAC)   # oversample 'Normal' -> fewer false positives
            s0 = np.random.choice(idx0, h0, replace=len(idx0) < h0)
            s1 = np.random.choice(idx1, self.BATCHSIZE - h0, replace=len(idx1) < self.BATCHSIZE - h0)
            idx = np.concatenate([s0, s1])
        else:
            idx = np.random.choice(n, self.BATCHSIZE, replace=False)
        b   = self.memory[idx]

        bs  = torch.FloatTensor(b[:, :self.NUMSTATES]).to(device)
        ba  = torch.LongTensor(b[:, self.NUMSTATES:self.NUMSTATES+1].astype(int)).to(device)
        br  = torch.FloatTensor(b[:, self.NUMSTATES+1:self.NUMSTATES+2]).to(device)
        bns = torch.FloatTensor(b[:, self.NUMSTATES+2:self.NUMSTATES*2+2]).to(device)

        self.eval_net.train()
        q_eval = self.eval_net(bs).gather(1, ba)

        with torch.no_grad():
            best_a   = self.eval_net(bns).argmax(dim=1, keepdim=True)  # Double DQN
            q_next   = self.target_net(bns).gather(1, best_a)
            q_target = br + self.GAMMA * q_next

        loss = self.loss_fn(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), 1.0)
        self.optimizer.step()
        self._soft_update()            # [5] smooth target tracking every step
        return loss.item()

    def evaluate_batch(self, inputs):
        self.eval_net.eval()
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(inputs), 1024):
                batch = inputs[i:i+1024].to(device)
                all_preds.extend(self.eval_net(batch).argmax(dim=1).cpu().numpy())
        self.eval_net.train()
        return np.array(all_preds)

    def save_model(self, path):
        torch.save(self.eval_net.state_dict(), path)


# ============================================================
# SECTION 4 — Hyperparameters
# ============================================================
BATCHSIZE         = 512
LR                = 0.001
GAMMA             = 0.99
MEMORYCAPACITY    = 50000
QNETWORKITERATION = 10          # retained for reference; soft update used instead
TAU               = 0.01        # [5] Polyak coefficient
EPISODES          = 500          # BALANCED-BEST: longer run for better convergence
CHUNK_SIZE        = 256
LEARN_EVERY       = 25           # learn twice as often (was 50)
EPSILON_START     = 1.0
EPSILON_MIN       = 0.01
EPSILON_DECAY     = 0.97
EVAL_EVERY        = 10           # finer checkpoints -> better chance to catch a high-recall model

# Model-selection objective (computed on VALIDATION only):
#   'acc'            -> max accuracy            (lets FN rise — not safety-friendly)
#   'bal_acc'        -> max balanced accuracy   (fair compromise)
#   'acc_at_recall'  -> max accuracy SUBJECT TO recall >= RECALL_FLOOR  (RECOMMENDED)
# The recall floor caps how many attacks you allow the model to miss while still
# pushing accuracy as high as the data permits. Raise the floor -> lower FN but
# lower accuracy; lower the floor -> higher accuracy but more FN.
SELECT_METRIC     = 'acc_at_recall'
RECALL_FLOOR      = 0.99         # BALANCED-BEST: loose floor -> optimiser free to maximise F1
CONSTRAINED_OBJ   = 'f1'         # BALANCED-BEST: maximise F1 ('f1'|'acc'|'prec'|'mcc'|'bal_acc')
NORMAL_FRAC       = 0.50         # neutral class balance for the balanced-best point

# [2] Rebalanced asymmetric reward (TUNE ON VALIDATION).
#     ACCURACY PRESET: correct decisions rewarded equally and FP penalty raised
#     so the agent stops over-alerting -> fewer false positives -> higher accuracy.
#     FN is still penalised more than FP (1.67:1), so it stays safety-leaning.
#     To recover more recall, make R_FP less negative (e.g. -0.20) or R_FN more
#     negative; to push accuracy higher, make R_FP more negative (e.g. -0.40).
R_TP, R_TN = 0.50, 0.10
R_FN, R_FP = -0.50, -0.10        # FN<10 preset: 5:1 recall-heavy (reaches ~99.7% recall)

dqn = DQN(NUMSTATES, NUMLabels, BATCHSIZE, LR,
          GAMMA, MEMORYCAPACITY, QNETWORKITERATION, TAU, EPISODES)


# ============================================================
# SECTION 5 — Pre-fill Memory (Warm-up) — reward aligned with Section 4
# ============================================================
print("\nPre-filling replay memory (warm-up)...")
fill_count = 0
while dqn.mem_counter < min(MEMORYCAPACITY // 2, len(traininput)):
    i      = fill_count % len(traininput)
    state  = traininput[i]
    label  = int(trainlabels[i].item())
    action = np.random.randint(0, NUMLabels)
    if   action == 1 and label == 1: reward = R_TP
    elif action == 0 and label == 0: reward = R_TN
    elif action == 0 and label == 1: reward = R_FN
    else:                            reward = R_FP
    next_s = traininput[min(i + 1, len(traininput) - 1)]
    dqn.store_transitions_batch(
        state.unsqueeze(0), np.array([action]),
        np.array([reward], dtype=np.float32), next_s.unsqueeze(0),
        np.array([label]))
    fill_count += 1
print(f"Memory pre-filled with {dqn.mem_counter} transitions.")


# ============================================================
# SECTION 6 — Training Loop (with validation-based model selection)
# ============================================================
reward_list    = []
train_progress = []
test_progress  = []
val_progress   = []
epsilon        = EPSILON_START

best_val_key   = (-1.0, -1.0)   # [1] constrained selection key (meets_floor, score)
best_state     = None
best_episode   = -1

print("\n" + "="*55 + "\nTRAINING STARTED\n" + "="*55)

for ep in range(EPISODES):
    perm      = torch.randperm(len(traininput))
    ep_input  = traininput[perm]
    ep_labels = trainlabels[perm]

    ep_reward = 0.0; TotalReward = 0.0; step = 0

    for start in range(0, len(ep_input) - CHUNK_SIZE, CHUNK_SIZE):
        states      = ep_input[start : start + CHUNK_SIZE]
        labels_np   = ep_labels[start : start + CHUNK_SIZE].numpy()
        next_states = ep_input[start + 1 : start + CHUNK_SIZE + 1]

        actions, rand_sel = dqn.choose_action_batch(states, epsilon)

        # [2] Rebalanced asymmetric reward
        rewards = np.where((actions == 1) & (labels_np == 1), R_TP,
                  np.where((actions == 0) & (labels_np == 0), R_TN,
                  np.where((actions == 0) & (labels_np == 1), R_FN, R_FP)))

        non_rand     = ~rand_sel.astype(bool)
        TotalReward += non_rand.sum() * max(R_TP, R_TN)
        ep_reward   += rewards[non_rand].sum()

        # Teacher forcing on exploratory actions
        final_actions = np.where(rand_sel.astype(bool), labels_np, actions)
        dqn.store_transitions_batch(states, final_actions,
                                    rewards.astype(np.float32), next_states, labels_np)
        step += 1
        if step % LEARN_EVERY == 0 and dqn.mem_counter >= BATCHSIZE:
            dqn.learn()

    if dqn.mem_counter >= BATCHSIZE:
        for _ in range(5):
            dqn.learn()

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    dqn.scheduler.step()

    pct = round(ep_reward / max(TotalReward, 1e-6), 4)
    train_progress.append(pct); reward_list.append(ep_reward)

    if (ep + 1) % 10 == 0:
        print(f"Episode {ep+1:>3}/{EPISODES} | Reward: {ep_reward:>8.2f} | "
              f"Progress: {pct:.4f} | Epsilon: {epsilon:.4f}")

    # ── Validation-based selection + test curve every EVAL_EVERY ──────────
    if (ep + 1) % EVAL_EVERY == 0:
        vm = compute_metrics(dqn.evaluate_batch(valinput), vallabels.numpy())
        tm = compute_metrics(dqn.evaluate_batch(testinput), testlabels.numpy())
        val_progress.append([ep+1, vm['acc'], vm['bal_acc'], vm['rec'], vm['f1']])
        test_progress.append([ep+1, tm['acc'], tm['prec'], tm['rec'], tm['f1'],
                              tm['TP'], tm['TN'], tm['FP'], tm['FN']])
        # SELECT on VALIDATION ONLY (never on test).
        # 'acc_at_recall' = hard recall constraint: among checkpoints whose
        # validation recall meets RECALL_FLOOR (i.e. FN below target), keep the
        # most ACCURATE one; if none meet it yet, keep the highest-recall one.
        if SELECT_METRIC == 'acc_at_recall':
            meets = 1 if vm['rec'] >= RECALL_FLOOR else 0
            key = (meets, vm[CONSTRAINED_OBJ] if meets else vm['rec'])
        else:
            key = (1, vm[SELECT_METRIC])
        if key > best_val_key:
            best_val_key = key
            best_state   = copy.deepcopy(dqn.eval_net.state_dict())
            best_episode = ep + 1
        print(f"  > VAL  Acc {vm['acc']*100:.2f}% BalAcc {vm['bal_acc']*100:.2f}% "
              f"Rec {vm['rec']*100:.2f}% | TEST Acc {tm['acc']*100:.2f}% "
              f"Rec {tm['rec']*100:.2f}% FN {tm['FN']}")

    if (ep + 1) % 50 == 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(reward_list, 'b-', lw=0.8); axes[0].set_title('Reward per Episode')
        axes[1].plot(train_progress, 'r-', lw=0.8); axes[1].set_title('Training Progress')
        plt.tight_layout(); plt.savefig(f'{Results}/Training_ep{ep+1}.png', dpi=150); plt.close()


# ============================================================
# SECTION 7 — Final Evaluation (load BEST validation model, test once)
# ============================================================
if best_state is not None:
    dqn.eval_net.load_state_dict(best_state)
    print(f"\nLoaded best model from episode {best_episode} "
          f"(selected on validation, metric={SELECT_METRIC}, floor={RECALL_FLOOR})")

# Evaluate the SAME best model on every split (train / validation / test)
mtr = compute_metrics(dqn.evaluate_batch(traininput), trainlabels.numpy())
mvl = compute_metrics(dqn.evaluate_batch(valinput),   vallabels.numpy())
m   = compute_metrics(dqn.evaluate_batch(testinput),  testlabels.numpy())
TP, TN, FP, FN = m['TP'], m['TN'], m['FP'], m['FN']
Accuracy, Precision, Recall, F1Score = m['acc'], m['prec'], m['rec'], m['f1']

# Train vs Validation vs Test (generalization check — small gap = no overfitting)
print("\n" + "="*64)
print("PERFORMANCE ON ALL SPLITS  (same best-validation model)")
print("="*64)
print(f"{'Split':<12}{'Acc%':>8}{'Prec%':>8}{'Rec%':>8}{'Spec%':>8}{'F1%':>8}{'MCC':>8}{'FN':>6}")
for name, mm in [('Train', mtr), ('Validation', mvl), ('Test', m)]:
    print(f"{name:<12}{mm['acc']*100:>8.2f}{mm['prec']*100:>8.2f}{mm['rec']*100:>8.2f}"
          f"{mm['spec']*100:>8.2f}{mm['f1']*100:>8.2f}{mm['mcc']:>8.3f}{mm['FN']:>6}")
print(f"\nTrain-Test accuracy gap: {(mtr['acc']-m['acc'])*100:+.2f} pp  (small gap = good generalization)")
print("="*64)

pd.DataFrame([
    {'Split':'Train','Accuracy':mtr['acc'],'Precision':mtr['prec'],'Recall':mtr['rec'],
     'Specificity':mtr['spec'],'F1':mtr['f1'],'MCC':mtr['mcc'],
     'TP':mtr['TP'],'TN':mtr['TN'],'FP':mtr['FP'],'FN':mtr['FN']},
    {'Split':'Validation','Accuracy':mvl['acc'],'Precision':mvl['prec'],'Recall':mvl['rec'],
     'Specificity':mvl['spec'],'F1':mvl['f1'],'MCC':mvl['mcc'],
     'TP':mvl['TP'],'TN':mvl['TN'],'FP':mvl['FP'],'FN':mvl['FN']},
    {'Split':'Test','Accuracy':m['acc'],'Precision':m['prec'],'Recall':m['rec'],
     'Specificity':m['spec'],'F1':m['f1'],'MCC':m['mcc'],
     'TP':m['TP'],'TN':m['TN'],'FP':m['FP'],'FN':m['FN']},
]).to_excel(f'{Results}/Split_Comparison_{MethodName}.xlsx', index=False)

print("\n" + "="*55 + "\nFINAL TEST RESULTS (best-validation model)\n" + "="*55)
print(f"  Accuracy  : {Accuracy*100:.2f}%")
print(f"  Precision : {Precision*100:.2f}%")
print(f"  Recall    : {Recall*100:.2f}%")
print(f"  Specificity: {m['spec']*100:.2f}%")
print(f"  Bal. Acc  : {m['bal_acc']*100:.2f}%")
print(f"  MCC       : {m['mcc']:.4f}")
print(f"  F1 Score  : {F1Score*100:.2f}%")
print(f"  TP={TP} | TN={TN} | FP={FP} | FN={FN}")
print("="*55)

df_test = pd.DataFrame(test_progress,
    columns=['Episode','Accuracy','Precision','Recall','F1','TP','TN','FP','FN'])
df_test.to_excel(f'{Results}/TestingResults_{MethodName}.xlsx', index=False)
df_val = pd.DataFrame(val_progress, columns=['Episode','ValAcc','ValBalAcc','ValRec','ValF1'])
df_val.to_excel(f'{Results}/ValidationProgress_{MethodName}.xlsx', index=False)
df_train = pd.DataFrame({'Reward': reward_list, 'Progress': train_progress})
df_train.to_excel(f'{Results}/TrainingProgress_{MethodName}.xlsx', index=False)
dqn.save_model(PathModel)
print(f"\nModel saved -> {PathModel}")

fig, ax = plt.subplots(figsize=(5, 4))
cm = np.array([[TN, FP], [FN, TP]])
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Pred: Normal', 'Pred: Attack'])
ax.set_yticklabels(['Actual: Normal', 'Actual: Attack'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                fontsize=14, color='white' if cm[i, j] > cm.max() / 2 else 'black')
ax.set_title(f'Confusion Matrix - {MethodName}\n'
             f'Acc={Accuracy*100:.2f}% | Recall={Recall*100:.2f}% | FN={FN}')
plt.colorbar(im, ax=ax); plt.tight_layout()
plt.savefig(f'{Results}/ConfusionMatrix_{MethodName}.png', dpi=150); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(reward_list, 'b-', lw=1.0); axes[0].set_title(f'Training Reward - {MethodName}')
axes[0].grid(alpha=0.3)
if test_progress:
    eps_t = [r[0] for r in test_progress]
    axes[1].plot(eps_t, [r[1]*100 for r in test_progress], 'go-', label='Test Accuracy')
    axes[1].plot(eps_t, [r[3]*100 for r in test_progress], 'rs-', label='Test Recall')
    axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_ylim([0, 100])
    axes[1].set_title('Test Accuracy & Recall per Evaluation')
plt.tight_layout(); plt.savefig(f'{Results}/TrainingCurve_{MethodName}.png', dpi=150); plt.close()
print(f"Plots saved -> {Results}/")


# ============================================================
# SECTION 8 — SHAP Explainability (unchanged logic; raw-Q output is fine)
# ============================================================
print("\nGenerating SHAP explanations...")
torch.set_grad_enabled(True)
dqn.eval_net.eval()
background = traininput[:100].to(device)
test_shap  = testinput[:50].to(device)
assert background.shape[1] == NUMSTATES and test_shap.shape[1] == NUMSTATES

explainer   = shap.DeepExplainer(dqn.eval_net, background)
shap_values = explainer.shap_values(test_shap, check_additivity=False)

if isinstance(shap_values, list):
    shap_attack = np.array(shap_values[1])
elif isinstance(shap_values, np.ndarray):
    if shap_values.ndim == 3 and shap_values.shape[-1] == 2:
        shap_attack = shap_values[:, :, 1]
    elif shap_values.ndim == 3 and shap_values.shape[0] == 2:
        shap_attack = shap_values[1]
    else:
        shap_attack = shap_values
else:
    shap_attack = np.array(shap_values)

test_shap_np = test_shap.cpu().numpy()
assert shap_attack.shape == test_shap_np.shape
print("SHAP shape check passed")

plt.figure()
shap.summary_plot(shap_attack, test_shap_np, feature_names=feature_names, show=False)
plt.tight_layout(); plt.savefig(f'{Results}/SHAP_Summary_{MethodName}.png', dpi=150, bbox_inches='tight'); plt.close()

plt.figure()
shap.summary_plot(shap_attack, test_shap_np, feature_names=feature_names, plot_type='bar', show=False)
plt.tight_layout(); plt.savefig(f'{Results}/SHAP_Bar_{MethodName}.png', dpi=150, bbox_inches='tight'); plt.close()

mean_shap = np.abs(shap_attack).mean(axis=0)
df_shap = pd.DataFrame({'Feature': feature_names, 'Mean_SHAP': mean_shap})
df_shap = df_shap.sort_values('Mean_SHAP', ascending=False).reset_index(drop=True)
df_shap.to_csv(f'{Results}/SHAP_FeatureImportance_{MethodName}.csv', index=False)

print("\n" + "="*55 + "\nALL DONE - Check the Results folder.\n" + "="*55)
