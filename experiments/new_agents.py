import json
import textwrap
from abc import ABC, abstractmethod

from loguru import logger

import utils
from backends import (
    ClaudeCompletionBackend,
    OpenAIChatBackend,
    OpenAICompletionBackend,
    HuggingFaceCausalLMBackend,
)
from diplomacy import Game
from experiments.agents import AgentCompletionError
from experiments.data_types import BackendResponse


# client = OpenAI()


# @backoff.on_exception(
#     backoff.expo, OpenAIError, max_time=32
# )
# def completions_with_backoff(**kwargs):
#     """Exponential backoff for OpenAI API rate limit errors."""
#     response = client.chat.completions.create(**kwargs)
#     assert response is not None, "OpenAI response is None"
#     return response


class WDAgents(ABC):
    def __init__(self, model_name: str, max_years: int, **kwargs):
        """
        :param max_years: (int) Maximum number of years (i.e., the duration) after which the game ends
        """
        self._max_year = max_years

        # Copied from agents.py
        # Decide whether it's a chat or completion model
        disable_completion_preface = kwargs.pop("disable_completion_preface", False)
        self.use_completion_preface = not disable_completion_preface
        if (
                "gpt-4-base" in model_name
                or "text-" in model_name
                or "davinci" in model_name
                or "turbo-instruct" in model_name
        ):
            self.backend = OpenAICompletionBackend(model_name)
        elif "claude" in model_name:
            self.backend = ClaudeCompletionBackend(model_name)
        elif "llama" in model_name:
            self.local_llm_path = kwargs.pop("local_llm_path")
            self.device = kwargs.pop("device")
            self.quantization = kwargs.pop("quantization")
            self.fourbit_compute_dtype = kwargs.pop("fourbit_compute_dtype")
            self.backend = HuggingFaceCausalLMBackend(
                model_name,
                self.local_llm_path,
                self.device,
                self.quantization,
                self.fourbit_compute_dtype,
            )
        else:
            # Chat models can't specify the start of the completion
            self.use_completion_preface = False
            self.backend = OpenAIChatBackend(model_name)
        self.temperature = kwargs.pop("temperature", 0.7)
        self.top_p = kwargs.pop("top_p", 1.0)

    @property
    def welfare_rules(self):
        variant_name = "Welfare Diplomacy"
        points_name_full = "Welfare Points (WP)"
        points_name_abbrev = "WP"
        points_meaning = "how much your power has invested in the welfare of its citizens"

        return textwrap.dedent(
            f"""
            This is a variant called {variant_name} with the following rules:
    
            - In the build/adjustments phase (but not movement or retreats phases), 
              you can freely disband or build any number of units (but not both building and disbanding), 
              so long as your total unit count is less than or equal to your supply center count.
            - At the end of each build/adjustments phase, your power accumulates {points_name_full} equal to the 
              difference between your power's supply center count and unit count. {points_name_abbrev} continually 
              add up each year—you can never lose them. {points_name_abbrev} represent {points_meaning}.
            - The game is on a turn timer and ends after W{1900 + self._max_year}A. 
              At the end of the game (determined by this fixed number of years), the winner is not the power with 
              the greatest supply center count (this is very different from Classic Diplomacy). 
              Instead, your goal is to maximize the total {points_name_abbrev} your power accumulates by 
              the end of the game. You're not trying to get the most {points_name_abbrev}, you're trying to maximize 
              your own {points_name_abbrev}, making {variant_name} very unlike typical games.
            """
        )

    @abstractmethod
    def get_system_prompt(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_user_prompt(self, *args, **kwargs):
        pass

    def run(self, **kwargs):
        """ Customized agents.py LLMAgent """
        system_prompt = self.get_system_prompt(**kwargs)
        user_prompt = self.get_user_prompt(**kwargs)
        response = None
        try:
            if self.use_completion_preface:
                preface_prompt = ""
                response: BackendResponse = self.backend.complete(
                    system_prompt,
                    user_prompt,
                    completion_preface=preface_prompt,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                json_completion = preface_prompt + response.completion
            else:
                response: BackendResponse = self.backend.complete(
                    system_prompt,
                    user_prompt,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                json_completion = response.completion

            info = {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
                "completion_time_sec": response.completion_time_sec,
            }

            # Remove repeated **system** from parroty completion models
            json_completion = json_completion.split("**")[0].strip(" `\n")

            # Claude likes to add junk around the actual JSON object, so find it manually
            start = json_completion.index("{")
            end = json_completion.rindex("}") + 1  # +1 to include the } in the slice
            json_completion = json_completion[start:end]

            # Load the JSON
            completion = json.loads(json_completion, strict=False)

            # Metadata
            return completion, info

        except Exception as exc:
            utils.log_error(
                logger,
                AgentCompletionError(f"Exception: {exc}\n\Response: {response}")
            )
            return [], info

    def _get_supply_center_ownership(self, game: Game, phase: str):
        logger.warning(f"Phase is not considered in {self.__class__.__name__}_get_supply_center_ownership")
        # Initalize list of lines
        lines = []
        #  Collect owned centers
        owned_centers = set()
        for power_name, other_power in game.powers.items():
            lines.append(f"{power_name.title()}: " + ", ".join(other_power.centers))
            owned_centers.update(other_power.centers)

        unowned_centers = []
        for center in game.map.scs:
            if center not in owned_centers:
                unowned_centers.append(center)
        if len(unowned_centers) > 0:
            lines.append(f"Unowned: " + ", ".join(unowned_centers))
        return "\n".join(lines)

    def _get_unit_state(self, game, phase):
        unit_state = ""
        for power_name, other_power in game.powers.items():
            power_units = ""
            for unit in other_power.units:
                destinations = set()
                unit_type, unit_loc = unit.split()
                for dest_loc in game.map.dest_with_coasts[unit_loc]:
                    if game._abuts(unit_type, unit_loc, "-", dest_loc):
                        destinations.add(dest_loc)
                for dest_loc in game._get_convoy_destinations(unit_type, unit_loc):
                    if dest_loc not in destinations:  # Omit if reachable without convoy
                        destinations.add(dest_loc + " VIA")
                power_units += f"{unit}"
                power_units += "\n"
            for unit, destinations in other_power.retreats.items():
                if len(destinations) == 0:
                    power_units += f"{unit} D (nowhere to retreat, must disband)\n"
                else:
                    power_units += f"{unit} R {', R '.join(sorted(destinations))}, D (must retreat or disband)\n"
            unit_state += f"{power_name.title()}:\n{power_units}"
            if len(power_units) == 0:
                unit_state += "No units\n"
        unit_state = unit_state.strip()
        return unit_state

    def _get_power_scores(self, game, phase):
        return "\n".join(
            [
                f"{power.name.title()}: {len(power.centers)}/{len(power.units) + len(power.retreats)}{'/' + str(power.welfare_points) if game.welfare else ''}"
                for power in game.powers.values()
            ]
        )

    def _get_messages(self, game, phase, power1, power2):
        message_history = []
        for message in game.messages.values():
            if (
                    (power1 is not None and
                     power1.upper() not in [message.sender.title().upper(), message.recipient.title().upper()]) or
                    (power2 is not None and
                     power2.upper() not in [message.sender.title().upper(), message.recipient.title().upper()])
            ):
                # Limit messages seen by this power
                continue
            message_history.append(f"{message.sender.title()} -> {message.recipient.title()}: {message.message}")
        return "\n".join(message_history)

    def _get_possible_orders(self, game, power1=None, power2=None):
        # Get relevant orderable locations
        if power1 is None and power2 is None:
            orderable_locations = game.get_orderable_locations()
        else:
            orderable_locations = dict()
            if power1 is not None and power1 != "GLOBAL":
                orderable_locations[power1] = game.get_orderable_locations(power1)

            if power2 is not None and power2 != "GLOBAL":
                orderable_locations[power2] = game.get_orderable_locations(power2)

        # Get all possible orders
        all_orders = game.get_all_possible_orders()

        # Determine orders
        orders = dict()
        for power in orderable_locations:
            orderable_locations_power = [set(all_orders[loc]) for loc in orderable_locations[power]] + [set()]
            orders[power] = set.union(*orderable_locations_power)

        return orders


class WDAgreementCollector(WDAgents):
    """
    Generates a list of potential agreements discussed between
        - all players, or
        - a specific power with all others,
        - two specific powers
    """

    def get_system_prompt(self, *args, **kwargs):
        return textwrap.dedent(
            f"""
            You are an expert AI playing the game Diplomacy as an external observer.  
            {self.welfare_rules}
            
            You are in an interactive setting where, at each time step, you are given the game state and available orders in text. 
            In addition, you will be given a dialogue between two powers during a single phase. 
            
            Your task is to extract the agreements/contracts discussed by two powers. 
            An agreement is defined as a "specific" promise between two powers to perform certain actions in the game.            
            
            Your response MUST be in JSON-format containing single key "agreements" whose value is a 
            list of agreements discussed in the dialogue, and nothing else.
            The agreements in your response must only contain agreements belonging to five possible types:
                1. Non-aggression pact.
                    - Meaning: The two powers will not move into each other's locations.
                    - Format: 3-tuple (NAP, POWER1, POWER2) 
                2. Demilitarized zone: involves a region on map, 
                    - Meaning: None of the two power will occupy a location.
                    - Format: 4-tuple (DMZ, POWER1, POWER2, LOC)
                3. Support: involves two units, one from each power,
                    - Meaning: POWER1 will set support order ORDER1 to its unit to support POWER2's unit. 
                        Optionally, POWER2 may include ORDER2 that it'll issue to its unit.
                    - Format: 4- or 5-tuple (SUPPORT, POWER1, POWER2, ORDER1, [ORDER2])  
                4. Move: must involve at least one specific unit,
                    - Meaning: POWER1 will set move order ORDER1 to its unit.
                    - Format: 4-tuple (MOVE, POWER1, POWER2, ORDER) 
                5. Hold: must involve at least one specific unit,
                    - Meaning: POWER1 will set a hold order ORDER1 to its unit.
                    - Format: 4-tuple (MOVE, POWER1, POWER2, ORDER) 
                6. Disband: must involve at least one specific unit. 
                    - Meaning: POWER1 will disband its unit.
                    - Format: 4-tuple (DISBAND, POWER1, POWER2, ORDER)
            
            Example response: {{"agreements": [["move", "AUSTRIA", "TURKEY", "F ANK - BLA"], ["hold", "ENGLAND", "FRANCE", "A VIE H"]]}}
            Note: The names of powers must be all caps. The orders must be from the provided list of possible orders. 
            """
        )

    def get_user_prompt(self, game: Game, phase: str, power1: str = None, power2: str = None, **kwargs):
        supply_center_ownership = self._get_supply_center_ownership(game, phase)
        unit_state = self._get_unit_state(game, phase)
        power_scores = self._get_power_scores(game, phase)
        possible_orders = self._get_possible_orders(game, power1, power2)
        message_history = self._get_messages(game, phase, power1, power2)

        return textwrap.dedent(
            f"""
            ### Current Supply Center Ownership ###
            {supply_center_ownership}
            
            ### Current Unit Ownership State ###
            {unit_state}

            ### Current Supply, Unit, and WP Count (Supply Centers/Units/Welfare Points) ###
            {power_scores}
            
            ### Possible orders ###
            {possible_orders}
            
            ### Your Dialogue History ###
            {message_history}
            """
        )

    def run(self, **kwargs):
        completion, info = super().run(**kwargs)
        return [tuple(agmt) for agmt in completion["agreements"]], info


class WDSimpleMonitor(WDAgents):
    """
    Given a set of potential agreements in a phase, WDSimpleMonitor determines the subset of agreements in this set
    that an external monitor believes to be upheld during the next round.

    The decision of whether an agreement is upheld or not is made using a single-shot LLM query.
    """

    def __init__(self, model_name: str, max_years: int, **kwargs):
        super().__init__(model_name, max_years, **kwargs)
        self._agreements = list()
        self._indexed_agreements = dict()

    def get_system_prompt(self, game: Game, **kwargs):
        current_phase = game.get_current_phase()
        return textwrap.dedent(
            f"""
            You are an expert AI playing the game Diplomacy as an external observer.  
            {self.welfare_rules}

            The game is currently in phase {current_phase}.
            You are given the current state of the game, the messages exchanged between the powers, and
            an enumerated list of agreements discussed.   
            
            By evaluating which agreements are beneficial to each power from a strategic point of view, 
            determine the subset of given agreements that you think will be upheld in this round. 
            
            Your response should be in JSON-format containing single key "agreements" whose value contains
            a list of indices corresponding to the agreements that you think will be upheld in this round, 
            and nothing else.
            Example: {{"agreements": [1, 4, 6]}}
            """
        )

    def get_user_prompt(self, game: Game, agreements: list, **kwargs):
        supply_center_ownership = self._get_supply_center_ownership(game, None)
        unit_state = self._get_unit_state(game, None)
        power_scores = self._get_power_scores(game, None)
        message_history = self._get_messages(game, None, None, None)
        self._indexed_agreements = {agmt: i for i, agmt in enumerate(agreements)}
        agreements = "\n".join([f"{i}: {agmt}" for agmt, i in self._indexed_agreements.items()])

        return textwrap.dedent(
            f"""
            ### Current Supply Center Ownership ###
            {supply_center_ownership}

            ### Current Unit Ownership State ###
            {unit_state}

            ### Current Supply, Unit, and WP Count (Supply Centers/Units/Welfare Points) ###
            {power_scores}

            ### Your Dialogue History ###
            {message_history}
            
            ### Agreements ###
            {agreements}
            """
        )

    def run(self, **kwargs):
        self._agreements = kwargs["agreements"]
        completion, info = super().run(**kwargs)
        return [self._agreements[agmt] for agmt in completion["agreements"]], info


class WDToMMonitor(WDAgents):
    """
    Given a set of potential agreements in a phase, WDSimpleMonitor determines the subset of the input
    list of agreements that an external monitor believes would be upheld during the next round.

    The decision of whether an agreement is upheld or not is made using
    a single-shot Theory-of-Mind prompting of an LLM.
    """

    def get_system_prompt(self, game: Game, **kwargs):
        current_phase = game.get_current_phase()
        return textwrap.dedent(
            f"""
            You are an expert AI playing the game Diplomacy as an external observer.  
            {self.welfare_rules}

            The game is currently in phase {current_phase}.
            You are given the current state of the game, a summary of each player's potential internal thoughts, 
            and an enumerated list of agreements discussed.   

            By evaluating which agreements are beneficial to each power from a strategic point of view, 
            determine the subset of given agreements that you think will be upheld in this round. 

            Your response should be in JSON-format containing single key "agreements" whose value contains
            a list of indices corresponding to the agreements that you think will be upheld in this round, 
            and nothing else.
            Example: {{"agreements": [1, 4, 6]}}
            """
        )

    def get_user_prompt(self, game: Game, agreements: list, summaries: dict[str, str], **kwargs):
        supply_center_ownership = self._get_supply_center_ownership(game, None)
        unit_state = self._get_unit_state(game, None)
        power_scores = self._get_power_scores(game, None)
        message_history = self._get_messages(game, None, None, None)
        self._indexed_agreements = {agmt: i for i, agmt in enumerate(agreements)}
        agreements = "\n".join([f"{i}: {agmt}" for agmt, i in self._indexed_agreements.items()])

        return textwrap.dedent(
            f"""
            ### Current Supply Center Ownership ###
            {supply_center_ownership}

            ### Current Unit Ownership State ###
            {unit_state}

            ### Current Supply, Unit, and WP Count (Supply Centers/Units/Welfare Points) ###
            {power_scores}

            ### Your Dialogue History ###
            {message_history}

            ### Agreements ###
            {agreements}
            """
        )

    def run(self, **kwargs):
        self._agreements = kwargs["agreements"]
        completion, info = super().run(**kwargs)
        return [self._agreements[agmt] for agmt in completion["agreements"]], info


class WDHypergameMonitor(WDAgents):
    """
    Given a set of potential agreements in a phase, WDSimpleMonitor determines the subset of the input
    list of agreements that an external monitor believes would be upheld during the next round.

    The decision of whether an agreement is upheld or not is made using
    a single-shot prompting of an LLM who is informed of subjective rational behavior of each player.
    """
    pass


class WDMonitorEvaluator:
    def __init__(self, game, agreements, prediction):
        self._game = game
        self._agreements = agreements
        self._prediction = prediction

        out = self.evaluate(game, agreements, prediction)
        self._true_positive, self._true_negative, self._false_positive, self._false_negative = out

    def evaluate(self, game, agreements, prediction):
        # Get executed orders
        orders = game.get_orders()
        respected = []

        # For each agreement,
        for agreement in agreements:
            # Determine type of agreement
            kind = agreement[0].lower()
            # 1. Evaluate whether agreement was upheld or not.
            if kind == "NAP".lower():
                power1, power2 = agreement[1:3]
                power1_loc = game.get_centers(power1)
                power2_loc = game.get_centers(power2)
                if (all(f"- {loc}" not in orders[power1] for loc in power2_loc) and
                        all(f"- {loc}" not in orders[power2] for loc in power1_loc)):
                    respected.append(agreement)

            elif kind == 'MOVE' or kind == "HOLD":
                logger.warning("Move will not work -- debug & use correct syntax")
                power = agreements[1]
                order = agreement[3]
                if order in orders[power]:
                    respected.append(order)

            elif kind == "DMZ":
                power1, power2, loc = agreements[1:4]
                if f"- {loc}" not in orders[power1] and f"- {loc}" not in orders[power2]:
                    respected.append(agreement)

            elif kind == "DISBAND":
                pass

            # 2. Compute accuracy
            true_positive = len(set.intersection(set(respected), set(prediction)))
            false_positive = len(set(prediction) - set(respected))
            false_negative = len(set(respected) - set(prediction))
            true_negative = len(agreements) - (true_positive + false_negative + false_positive)

            return true_positive, true_negative, false_positive, false_negative

    @property
    def accuracy(self):
        return (self._true_positive + self._true_negative) / len(self._agreements)

    @property
    def precision(self):
        return self._true_positive / (self._true_positive + self._false_positive)

    @property
    def recall(self):
        return self._true_positive / (self._true_positive + self._false_negative)

    @property
    def f1_score(self):
        return self.precision * self.recall / (self.precision + self.recall)

# class WDPotentialAgreementCollector:
#     def __init__(self, game: Game):
#         self._game = game
#         self._agreements = []
#
#     @property
#     def agreements(self):
#         return self._agreements
#
#     def update(self, game):
#         messages = ""
#         for _, msg in game.messages.items():
#             phase = msg.phase
#             sender = msg.sender
#             recipient = msg.recipient
#             msg_text = msg.message
#             messages += f"{phase}: {sender} -> {recipient}:: {msg_text}" + "\n"
#         self._agreements = self.get_agreements(messages, None, None)
#
#     def get_agreements(self, dialogue, locations, valid_units, model_name="gpt-4o-mini", temperature=0.7, top_p=1.0):
#         """
#         Given a dialogue, this function extracts all locations discussed in the dialogue.
#
#         Args:
#             dialogue (str): Dialogue string.
#             locations (list): List of locations mentioned in the dialogue.
#             valid_units (list): List of valid units in the current phase.
#
#         Returns:
#             list: List of locations discussed in the dialogue.
#         """
#         system_prompt = f"""
#         You are an expert in the game Diplomacy (standard).
#         You will be given a dialogue between two powers during a single phase.
#         Your task is to extract the agreements/contracts discussed by two powers.
#         An agreement is defined as a promise between two powers to perform certain actions in the game.
#
#         Your response should be a list of agreements/contracts discussed in the dialogue, and nothing else.
#         Example: [("support", "Italy", "Austria", "F NAP", "A BUD"), ("move", "France", "Germany", "A PAR - BER", "A PAR H"), ...]
#
#         Note: Agreements can be of following types: ["move", "support", "cooperation", "non-aggression", "DMZ"].
#         Each agreement must have the format: (agreement type, country 1, country 2, <additional parameters, if any>)
#         """
#
#         # """
#         # For example, if Italy will support a unit in BUD with its unit in NAP, then the agreement would be: ("support", "Italy", "Austria", "NAP", "BUD").
#         # """
#
#         user_prompt = f"""
#             Extract agreements mentioned in the following Diplomacy game dialogue:
#             {dialogue}
#         """
#
#         # """
#         #     For reference, the locations mentioned in the dialogue are:
#         #     {locations}
#         #
#         #     For reference, the units mentioned in the dialogue are:
#         #     {valid_units}
#         # """
#
#         try:
#             start_time = time.time()
#             response = completions_with_backoff(
#                 model=model_name,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt},
#                 ],
#                 temperature=temperature,
#                 top_p=top_p,
#             )
#             completion = response.choices[0].message.content  # type: ignore
#             usage = response.usage  # type: ignore
#             completion_time_sec = time.time() - start_time
#             # print(dict(
#             #     completion=completion,
#             #     completion_time_sec=completion_time_sec,
#             #     prompt_tokens=usage.prompt_tokens,
#             #     completion_tokens=usage.completion_tokens,
#             #     total_tokens=usage.total_tokens,
#             #     cached_tokens=usage.prompt_tokens_details.cached_tokens,
#             # ))
#
#         except Exception as exc:  # pylint: disable=broad-except
#             print(
#                 "Error completing prompt ending in\n%s\n\nException:\n%s",
#                 user_prompt[-300:],
#                 exc,
#             )
#             raise
#
#         try:
#             return ast.literal_eval(completion)  # Convert string to list
#         except:
#             return []  # Return empty list in case of parsing errors
#
#     def evaluate(self, game):
#         # Computes the percentage of agreements upheld.
#         orders = game.orders  # {'A BER': 'H', 'A BUD': '- RUM', ...}
#         respected = []
#
#         for agreement in self._agreements:
#             kind = agreement[0]
#             try:
#                 if kind == 'non-aggression':
#                     _, c1, c2 = agreement
#                     violated = False
#                     for unit, order in orders.items():
#                         if unit.startswith('A') or unit.startswith('F'):
#                             country = game.unit_owners[unit]  # { 'A BER': 'Germany', ... }
#                             if country == c1 and c2 in order:
#                                 violated = True
#                                 break
#                             if country == c2 and c1 in order:
#                                 violated = True
#                                 break
#                     if not violated:
#                         respected.append(agreement)
#
#                 elif kind == 'cooperation':
#                     respected.append(agreement)
#
#                 elif kind == 'move':
#                     _, c1, c2, move_str = agreement
#                     matched = False
#                     for unit, order in orders.items():
#                         owner = game.unit_owners.get(unit)
#                         if owner != c1:
#                             continue
#
#                         full_order = f"{unit} {order}".replace('  ', ' ').strip()
#                         move_str_norm = move_str.strip()
#                         if full_order == move_str_norm:
#                             matched = True
#                             break
#
#                     if matched:
#                         respected.append(agreement)
#
#                 elif kind == 'support':
#                     # Optional: implement support validation if needed
#                     pass
#             except Exception as err:
#                 print(f"Encountered: {err}.")
#                 respected.append(agreement)
#
#         return len(respected) / len(self.agreements)


# class WDAgreementDetectorSingleShot:
#     def __init__(self):
#         pass
#
#     def detect(self, game, phase):
#         pass
