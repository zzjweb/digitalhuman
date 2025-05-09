# Sentient Agent as a Judge
## Main Result


<table>
<tbody>
    <tr>
        <td colspan="2">Model</th><td colspan="2">Arena </th><td colspan="2">Sentient</th><td colspan="2">Supportive Dialogue </th>
    <tr>
    <tr>
        <td>Name</td>
        <td>Date</td>
        <td>Rank</td>
        <td>Score</td>
        <td>Rank</td>
        <td>Score</td>
        <td>Success</td>
        <td>Failure</td>
    </tr>
    <tr>
        <td>Gemini2.5-Pro</td>
        <td>2025-03-25</td>
        <td>1</td>
        <td>1439</td>
        <td>4</td>
        <td>62.9</td>
        <td>34</td>
        <td>25</td>
    </tr>
  <tr>
    <td>o3</td>
    <td>2025-04-16</td>
    <td>2</td>
    <td>1418</td>
    <td>5</td>
    <td>62.7</td>
    <td>32</td>
    <td>14</td>
</tr>

<tr>
    <td>GPT-4o-Latest</td>
    <td>2025-03-26</td>
    <td>2</td>
    <td>1408</td>
    <td>1</td>
    <td>79.9</td>
    <td>51</td>
    <td>4</td>
</tr>

<tr>
    <td>Gemini2.5-Flash-Think</td>
    <td>2025-04-17</td>
    <td>3</td>
    <td>1393</td>
    <td>3</td>
    <td>65.9</td>
    <td>35</td>
    <td>19</td>
</tr>

<tr>
    <td>GPT-4.5-Preview</td>
    <td>2025-02-27</td>
    <td>4</td>
    <td>1398</td>
    <td>6</td>
    <td>62.7</td>
    <td>23</td>
    <td>15</td>
</tr>

<tr>
    <td>Gemini2.0-Flash-Think</td>
    <td>2025-02-06</td>
    <td>7</td>
    <td>1380</td>
    <td>7</td>
    <td>62.3</td>
    <td>23</td>
    <td>23</td>
</tr>

<tr>
    <td>DeepSeek-V3-0324</td>
    <td>2025-03-24</td>
    <td>7</td>
    <td>1373</td>
    <td>10</td>
    <td>54.4</td>
    <td>19</td>
    <td>23</td>
</tr>

<tr>
    <td>GPT-4.1</td>
    <td>2025-04-14</td>
    <td>9</td>
    <td>1363</td>
    <td>2</td>
    <td>68.2</td>
    <td>35</td>
    <td>13</td>
</tr>

<tr>
    <td>DeepSeek-R1</td>
    <td>2025-01-21</td>
    <td>10</td>
    <td>1358</td>
    <td>11</td>
    <td>53.7</td>
    <td>31</td>
    <td>28</td>
</tr>

<tr>
    <td>Gemini2.0-Flash</td>
    <td>2025-02-06</td>
    <td>10</td>
    <td>1354</td>
    <td>15</td>
    <td>32.9</td>
    <td>8</td>
    <td>45</td>
</tr>

<tr>
    <td>o4-mini</td>
    <td>2025-04-16</td>
    <td>10</td>
    <td>1351</td>
    <td>13</td>
    <td>35.9</td>
    <td>10</td>
    <td>48</td>
</tr>

<tr>
    <td>o1</td>
    <td>2024-12-17</td>
    <td>12</td>
    <td>1350</td>
    <td>17</td>
    <td>29.0</td>
    <td>5</td>
    <td>51</td>
</tr>

<tr>
    <td>DeepSeek-V3</td>
    <td>2024-12-27</td>
    <td>18</td>
    <td>1318</td>
    <td>12</td>
    <td>37.6</td>
    <td>5</td>
    <td>39</td>
</tr>

<tr>
    <td>Claude3.7-Think</td>
    <td>2025-02-24</td>
    <td>21</td>
    <td>1301</td>
    <td>8</td>
    <td>61.3</td>
    <td>23</td>
    <td>19</td>
</tr>

<tr>
    <td>Claude3.7</td>
    <td>2025-02-24</td>
    <td>30</td>
    <td>1292</td>
    <td>9</td>
    <td>54.8</td>
    <td>19</td>
    <td>24</td>
</tr>

<tr>
    <td>GPT-4o</td>
    <td>2024-08-06</td>
    <td>45</td>
    <td>1265</td>
    <td>16</td>
    <td>31.8</td>
    <td>7</td>
    <td>51</td>
</tr>

<tr>
    <td>Llama3.3-70B</td>
    <td>2024-12-06</td>
    <td>56</td>
    <td>1256</td>
    <td>14</td>
    <td>33.3</td>
    <td>7</td>
    <td>47</td>
</tr>

<tr>
    <td>Qwen2.5-72B</td>
    <td>2024-09-19</td>
    <td>56</td>
    <td>1257</td>
    <td>18</td>
    <td>19.1</td>
    <td>4</td>
    <td>70</td>
</tr>
</table>

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


