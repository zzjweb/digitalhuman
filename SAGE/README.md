# Sentient Agent as a Judge


## Getting Started

### Step 1: Set Up Your LLM API

Configure your LLM API in the following files:

- **`sentient-agent-as-a-judge/simulator_response.py`**: Modify the `call_llm()` function.
- **`sentient-agent-as-a-judge/npc_response.py`**: Modify the `call_npc()` function.

### Step 2: Run the Code

Test the LLM set in **`npc_response`** with the LLM-based simulator set in **`simulator_response`**. Use our preset simulator profiles located at **`sentient-agent-as-a-judge/profile/simulator_profile_withfirsttalk.jsonl`**.

To execute the test, run the following command:

```python3
sentient-agent-as-a-judge/test_npc.py
```

## Prepare Simulator Profiles

If you wish to generate your own simulator profiles, follow these steps:

### Step 1: Build Seed Sets

Create seed talking sets and seed topic sets for generating profiles. These should include typical conversations and topics relevant to a scene.

Example Talking: 今天去公园了，真开心！

Example Topic: 在学校成绩总是不好怎么办

### Step 2: Set Up Your LLM API

Configure your LLM API in the following file:

- **`sentient-agent-as-a-judge/profile/build_profile.py`**: Modify the `call_llm()` function.

### Step 3: Build Profiles Without First Talk

Run the following command to build profiles without the first talk:
```python3
sentient-agent-as-a-judge/profile/build_profile.py
```

### Step 4: Build Profiles With First Talk

Set the profile path and store file path in **`sentient-agent-as-a-judge/build_profile_withfirsttalk.py`**. Then, run the following command:
```python3
sentient-agent-as-a-judge/build_profile_withfirsttalk.py
```


