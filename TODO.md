# Experiments for final paper 

## Baseline with different models. 
For now, models used are 
* gpt-4o-mini
* o1-mini (better at complex, math reasoning tasks, claimed by openai)
* ?? 
* ??

Agents to test:
* type-2: Switch policy @ year 2
* type-12: Switch with random disband @ year 2
* type-22: Switch with smart disband @ year 2
* type-32: Exploiter policy

For each experiment, we will run 10 times; to collect mean and variance.

# OLD
**WDMonitor Agent**
* [OK] Plug in an agent that prints messages as soon as they are sent.
* [CANCELED] Modify this agent to print messages + the locations discussed in that message (in context with history)
* Modify this agent to detect potential agreements.
* Can we distinguished signed agreements? If yes, check accuracy of detection.  