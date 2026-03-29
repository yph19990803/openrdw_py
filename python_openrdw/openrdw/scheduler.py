from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .geometry import Vector2
from .models import AgentState, Environment, GainsConfig, StepTrace
from .simulator import OpenRDWSimulator


@dataclass
class ScheduledAgent:
    agent_id: str
    state: AgentState
    environment: Environment
    gains: GainsConfig
    redirector: object
    resetter: object
    waypoints: list[Vector2]
    sampling_intervals: list[float] | None = None
    trace: list[StepTrace] = field(default_factory=list)


class MultiAgentScheduler:
    def __init__(self, agents: list[ScheduledAgent]):
        self.agents = agents
        self.simulators = {
            agent.agent_id: OpenRDWSimulator(
                environment=agent.environment,
                gains=agent.gains,
                redirector=agent.redirector,
                resetter=agent.resetter,
                waypoints=agent.waypoints,
                state=agent.state,
                sampling_intervals=agent.sampling_intervals,
                trace=agent.trace,
            )
            for agent in agents
        }

    def step(self, step_index: int, manual_inputs: dict[str, dict[str, bool]] | None = None) -> None:
        for agent in self.agents:
            self.simulators[agent.agent_id].prepare_frame()
        ordered = self._priority_order()
        movement_peer_states = copy.deepcopy([agent.state for agent in self.agents])
        for agent in ordered:
            manual_input = None if manual_inputs is None else manual_inputs.get(agent.agent_id)
            self.simulators[agent.agent_id].advance_movement(movement_peer_states, manual_input=manual_input)
        redirection_peer_states = copy.deepcopy([agent.state for agent in self.agents])
        for agent in ordered:
            self.simulators[agent.agent_id].apply_redirection_phase(redirection_peer_states)
        for agent in self.agents:
            self.simulators[agent.agent_id].finalize_frame(step_index)

    def run(self, steps: int) -> list[ScheduledAgent]:
        for step_index in range(steps):
            self.step(step_index)
        return self.agents

    def _priority_order(self) -> list[ScheduledAgent]:
        priorities: list[tuple[float, ScheduledAgent]] = []
        peer_states = [agent.state for agent in self.agents]
        for index, agent in enumerate(self.agents):
            if hasattr(agent.redirector, "get_priority"):
                priority = agent.redirector.get_priority(agent.state, agent.environment, agent.gains, peer_states)
                agent.state.priority = priority
            elif agent.state.priority == 0.0:
                agent.state.priority = float(index)
            priorities.append((agent.state.priority, agent))
        priorities.sort(key=lambda item: item[0], reverse=True)
        return [agent for _, agent in priorities]
