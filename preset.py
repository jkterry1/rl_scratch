import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.agents import PPO, PPOTestAgent
from all.approximation import VNetwork, Identity
from all.logging import DummyWriter
from all.optim import LinearScheduler
from all.policies import GaussianPolicy
from all.presets.builder import preset_builder
from all.presets.preset import Preset
from all.agents.multi.independent import IndependentMultiagent

default_hyperparameters = {
    # Common settings
    "discount_factor": 0.99,
    # Adam optimizer settings
    "lr": 2.5e-4,  # Adam learning rate
    "eps": 1e-5,  # Adam stability
    # Loss scaling
    "entropy_loss_scaling": 0.01,
    "value_loss_scaling": 0.5,
    # Training settings
    "clip_grad": 0.5,
    "clip_initial": 0.1,
    "clip_final": 0.01,
    "epochs": 4,
    "minibatches": 4,
    # Batch settings
    "n_envs": 4,
    "n_steps": 128,
    # GAE settings
    "lam": 0.95,
    # Model construction
    "value_model_constructor": 0,
    "policy_model_constructor": 0,
}


class PPOContinuousPreset(Preset):
    """
    Proximal Policy Optimization (PPO) Continuous Control Preset.

    Args:
        env (all.environments.GymEnvironment): The classic control environment for which to construct the agent.
        device (torch.device, optional): the device on which to load the agent

    Keyword Args:
        discount_factor (float): Discount factor for future rewards.
        lr (float): Learning rate for the Adam optimizer.
        eps (float): Stability parameters for the Adam optimizer.
        entropy_loss_scaling (float): Coefficient for the entropy term in the total loss.
        value_loss_scaling (float): Coefficient for the value function loss.
        clip_grad (float): Clips the gradient during training so that its L2 norm (calculated over all parameters) 
        # is no greater than this bound. Set to 0 to disable.
        clip_initial (float): Value for epsilon in the clipped PPO objective function at the beginning of training.
        clip_final (float): Value for epsilon in the clipped PPO objective function at the end of training.
        epochs (int): Number of times to literature through each batch.
        minibatches (int): The number of minibatches to split each batch into.
        n_envs (int): Number of parallel actors.
        n_steps (int): Length of each rollout.
        lam (float): The Generalized Advantage Estimate (GAE) decay parameter.
        value_model_constructor (function): The function used to construct the neural value model.
        policy_model_constructor (function): The function used to construct the neural policy model.
    """

    def __init__(self, env, device="cuda", **hyperparameters):
        hyperparameters = {**default_hyperparameters, **hyperparameters}
        super().__init__(hyperparameters["n_envs"])
        self.value_model = hyperparameters["value_model_constructor"].to(device)
        self.policy_model = hyperparameters["policy_model_constructor"].to(device)
        self.device = device
        self.action_space = env.action_spaces['piston_0']
        self.hyperparameters = hyperparameters
        self.agent_names = env.agents

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        n_updates = train_steps * self.hyperparameters['epochs'] * self.hyperparameters['minibatches'] / (self.hyperparameters['n_steps'] * self.hyperparameters['n_envs'])

        value_optimizer = Adam(self.value_model.parameters(), lr=self.hyperparameters['lr'], eps=self.hyperparameters['eps'])
        policy_optimizer = Adam(self.policy_model.parameters(), lr=self.hyperparameters['lr'], eps=self.hyperparameters['eps'])

        features = Identity(self.device)

        v = VNetwork(
            self.value_model,
            value_optimizer,
            loss_scaling=self.hyperparameters['value_loss_scaling'],
            clip_grad=self.hyperparameters['clip_grad'],
            writer=writer,
            scheduler=CosineAnnealingLR(
                value_optimizer,
                n_updates
            ),
        )

        policy = GaussianPolicy(
            self.policy_model,
            policy_optimizer,
            self.action_space,
            clip_grad=self.hyperparameters['clip_grad'],
            writer=writer,
            scheduler=CosineAnnealingLR(
                policy_optimizer,
                n_updates
            ),
        )

        def make_agent():
            PPO(
                features,
                v,
                policy,
                epsilon=LinearScheduler(
                    self.hyperparameters['clip_initial'],
                    self.hyperparameters['clip_final'],
                    0,
                    n_updates,
                    name='clip',
                    writer=writer
                ),
                epochs=self.hyperparameters['epochs'],
                minibatches=self.hyperparameters['minibatches'],
                n_envs=self.hyperparameters['n_envs'],
                n_steps=self.hyperparameters['n_steps'],
                discount_factor=self.hyperparameters['discount_factor'],
                lam=self.hyperparameters['lam'],
                entropy_loss_scaling=self.hyperparameters['entropy_loss_scaling'],
                writer=writer,
            )
        return IndependentMultiagent({
            agent_id: make_agent()
            for agent_id in self.agent_names
        })

    def test_agent(self):
        def make_agent():
            policy = GaussianPolicy(copy.deepcopy(self.policy_model), space=self.action_space)
            return PPOTestAgent(Identity(self.device), policy)
        return IndependentMultiagent({
            agent_id: make_agent()
            for agent_id in self.agent_names
        })


ppo = preset_builder('ppo', default_hyperparameters, PPOContinuousPreset)
