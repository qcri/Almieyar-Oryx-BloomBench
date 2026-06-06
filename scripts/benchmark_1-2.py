import csv
import uuid
import json
import os
import requests
import google.generativeai as genai
import re
from pathlib import Path
from PIL import Image


# --- Gemini Model Initialization ---
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3-pro-preview")


def generate_search_scenarios(hierarchy: dict, prompt_template: str) -> dict:
   """
   Takes the taxonomy level and a prompt template.
   Calls a text-generation LLM (Gemini) to generate 10 creative search queries and corresponding keywords.
   Returns a dictionary with "scenarios" and "keywords".
   """
   print("Generating search scenarios...")

   hierarchy_str = json.dumps(hierarchy, indent=2)

   prompt = prompt_template.replace("{{leaf}}", hierarchy['leaf'])
   prompt = prompt.replace("{{path}}", f"{hierarchy['lvl1']} -> {hierarchy['leaf']}")
   bloom_node = hierarchy['lvl1'].split('->')[0].strip()
   prompt = prompt.replace("{{bloom_node}}", bloom_node)
   prompt = prompt.replace("{{description}}", hierarchy['description'])
   print(prompt)

   try:
      response = model.generate_content([prompt])

      json_text = response.text.strip().lstrip("```json").rstrip("```")
      scenarios_data = json.loads(json_text)
      print("Successfully generated scenarios.")
      return scenarios_data
   except Exception as e:
      print(f"Error generating search scenarios: {e}")
      return {}



def generate_scenario(leaf):

        scenario_prompt_template = '''
You are generating Google Image Search queries intended to retrieve high-quality, language-independent images that can be used to create visual reasoning and visual understanding questions.

The images should depict clear, realistic scenes with minimal or no readable text.
Each image must make it easy to ask visual questions that test a model’s ability to understand, identify, differentiate, compare, or analyze visual elements in the scene.

The goal is not storytelling, but producing concrete, searchable visual scenarios that reliably return images suitable for evaluating visual cognition.

---

Task Context

Each task is grounded in Bloom’s Taxonomy and focuses on evaluating a specific visual capability.
You will be given a core visual concept and a cognitive level, along with a task description that explains what kind of visual understanding is being evaluated.

The generated images must naturally support question generation that evaluates the described task.

---

Input Fields

You will receive the following inputs:

- {{path}}: The full hierarchical classification of the task, used to infer domain and intent.
- {{leaf}}: The core visual concept that must be clearly represented in the image.
- {{bloom_node}}: The Bloom’s Taxonomy level defining the cognitive operation to support.
- {{description}}: A detailed description of the visual capability being evaluated. This is the primary guidance for what visual features, distinctions, or structures must be present.

Use description to ensure the image:
- Contains visually distinct elements relevant to the task
- Allows fine-grained recognition or differentiation
- Avoids ambiguity caused by background clutter or irrelevant objects

---


Bloom's Taxonomy Levels & Abilities

1. **Remembering** – Recall facts and basic concepts  
   - *Abilities*: Recognizing, Recalling, Identifying, Listing

2. **Understanding** – Demonstrate comprehension of meaning  
   - *Abilities*: Explaining, Summarizing, Interpreting, Classifying

3. **Applying** – Use knowledge in new situations  
   - *Abilities*: Executing, Implementing, Using, Solving

4. **Analyzing** – Break information into parts to understand relationships  
   - *Abilities*: Comparing, Organizing, Deconstructing, Attributing

5. **Evaluating** – Make judgments based on criteria  
   - *Abilities*: Critiquing, Justifying, Validating, Supporting

6. **Creating** – Generate new ideas, artifacts, or compositions  
   - *Abilities*: Designing, Constructing, Planning, Producing


---

Scenario Generation Guidelines

1. Visually Concrete Scenes  
   Describe a single, realistic snapshot suitable for image search.

2. Leaf-Centered Visuals  
   The {{leaf}} must be clearly visible and visually distinguishable.

3. Bloom Alignment  
   The scene must enable the cognitive action implied by {{bloom_node}}.

4. Description-Driven Design  
   Scenarios must directly support the visual task described in {{description}}.

5. Minimal or No Text  
   Avoid readable text, labels, or signage.

6. Distinct Visual Features  
   Favor contrast, structure, or components that support fine-grained visual reasoning.

7. Cultural Coverage  
   Include Western, MENA, and Arabic contexts (at least one scenario each).

8. Environmental Diversity  
   Vary settings such as domestic, public, professional, outdoor, or technical spaces.

9. Conciseness  
   Each scenario must be under 20 words.

---

Expected Output

Return only a **valid JSON object** with two keys:
1.  `"scenarios"`: A list of **exactly 10 strings**, where each string is a descriptive scenario.
2.  `"keywords"`: A list of **exactly 10 strings**, where each string is a Google Image search query corresponding to the scenario at the same index.

Do **not** include any explanations, markdown, or formatting.

---

Example Output for "Visual Commonsense Reasoning" Task.

```json
{
  "scenarios": [
    "Man crossing street while looking at smartphone.",
    "Dog wearing birthday hat sitting at party table.",
    "Child pulling suitcase alone through airport terminal.",
    "Woman holding open umbrella inside office lobby.",
    "Cat pushing glass off kitchen counter.",
    "Camel standing amid busy city traffic.",
    "Family barbecuing on rooftop during heavy snowfall.",
    "Arab man pouring mint tea at outdoor market.",
    "Child asleep on pile of books in library.",
    "Hijabi woman taking selfie near seaside cliff."
  ],
  "keywords": [
    "man crossing street looking at phone",
    "dog birthday hat at table",
    "child suitcase airport terminal",
    "woman umbrella indoors office",
    "cat pushing glass off counter",
    "camel city street traffic",
    "winter rooftop barbecue snow",
    "Arab man pouring mint tea market",
    "child sleeping on books library",
    "hijab woman selfie cliff"
  ]
}
```

'''

        # 4. Generate search scenarios
        scenarios_data = generate_search_scenarios(leaf, scenario_prompt_template)
        if not scenarios_data:
            print("No scenarios were generated. Exiting.")
            return
        
        # 5. Save scenarios to file (ensure directory exists and sanitize filename)
        scenarios_dir = Path("scenarios")
        scenarios_dir.mkdir(parents=True, exist_ok=True)
        safe_name = f"{leaf['lvl1']} - {leaf['leaf']}"
        # Replace characters that are unsafe in filenames
        safe_name = re.sub(r'[<>:\\"/\\|?*\n\r]+', '_', safe_name).strip()
        scenarios_file = scenarios_dir / f"{safe_name}.json"
        try:
            with open(scenarios_file, "w", encoding="utf-8") as f:
                json.dump(scenarios_data, f, indent=4, ensure_ascii=False)
            print(f"Scenarios saved to {scenarios_file}")
        except Exception as e:
            print(f"Failed to save scenarios to {scenarios_file}: {e}")

        # print(f"\\nProcess complete. Dataset saved to {output_file}")

from tqdm import tqdm
# def main():
    # Read from CSV file
with open('Taxonomy.csv', newline='', encoding='utf-8') as csvfile:
   reader = csv.reader(csvfile)
   taxonomy_list = []
   for row in reader:
      if not row:
         continue
      # skip header rows where first cell contains 'hierarchy' or similar
      first = row[0].strip().lower()
      if 'hierarchy' in first or 'lvl' in first or first.startswith('#'):
         continue
      # expected columns: hierarchy_path, leaf, description, example
      hierarchy_path = row[0].strip()
      leaf = row[1].strip() if len(row) > 1 else ''
      description = row[2].strip() if len(row) > 2 else ''
      # If there are extra commas, join the remaining columns into example
      example = ','.join(col.strip() for col in row[3:]) if len(row) > 3 else ''
      taxonomy_list.append({
         "lvl1": hierarchy_path,
         "leaf": leaf,
         "description": description,
         "example": example
      })
# for i in tqdm(taxonomy_list):
#     # print(i)
#     generate_scenario(i)
#     # break


# if __name__ == "__main__":
#     main()
import threading
from tqdm import tqdm

# progress bar and lock for thread-safe updates
pbar = tqdm(total=len(taxonomy_list), desc="Generating scenarios")
pbar_lock = threading.Lock()

def worker(rank, total_threads):
   for i in range(rank, len(taxonomy_list), total_threads):
      try:
         generate_scenario(taxonomy_list[i])
      except Exception as e:
         print(f"Error in thread {rank} on {taxonomy_list[i]}: {e}")
      finally:
         try:
            with pbar_lock:
               pbar.update(1)
         except Exception:
            pass

num_threads = 5
threads = []

for rank in range(num_threads):
   t = threading.Thread(target=worker, args=(rank, num_threads))
   t.start()
   threads.append(t)

for t in threads:
   t.join()

# close progress bar
pbar.close()
