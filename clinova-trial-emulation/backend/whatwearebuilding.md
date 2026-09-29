# Pitch

# One-Liner

# High Level Usage

Doctors have uncertainty on some patient related questions. Pharmaceutical companies will use our software so doctors can reduce their uncertainty on such questions [find impact / purpose].

Our software utilizes a proven framework: virtual clinical trials (target trial emulation) to reduce bias / uncertainty. Each question and its results are logged in a database of the software. 
``` (only for future reference)
The goal is to develop technology which allows policy developers and makers to virtually simulate policy interventions fast with low effort, optional domain expertise, and provable clinical accuracy. We aim to achieve this goal by translating a framework called “Target Trial Emulation” (TTE) from academic labs, where the technology is currently experiencing exponentially growing popularity, into the hands of non-expert policymakers who need not be experts in clinical trial design, coding, or data analysis

to perform such analysis. The TTE is powerful because it empowers policymakers and researchers to iterate interventions fast, cheaply, and with clinical grade accuracy by enabling simulation of real world interventions of any scale on computers. Our team has already validated this technology for use in national and decade-scale drug safety policy development and gained access to giant medical databases for use during development.
```
X resists iteration.

# Methodology

We have two entities: the doctor and the pharmaceutical company. Our software is two faced. Let’s begin with the pharmaceutical company. 

For each drug, the pharmaceutical company trains a model for each question AND endpoint. We will focus on one particular drug and two main questions: safety and effectiveness. 

This model is trained via taking the [safety,effectiveness] and transforming it into a causal question that a TTE can answer while keeping the same regulatory / medical conclusion interpretation. Training follows the following pipeline:
Question is sent to Agent 1
I: “How safe is this drug?” O: “We emulate a target trial to determine the effect of [drug] on safety”
The model’s parameters in relation to a specific combination of (drug, question) are found through machine learning (gradient descent / XGBoost). Question is either “how safe” or “how effective”
Output is sent to a loop:
Agent 2a: Takes Causal Question + Shared Context from Agent 1 and generates design spec to design clinical trial
Agent 2b: Verified the spec + gives feedback based on search over relevant literature + clinical trials
Iterate until given greenlight once ready. Output: complete design spec for TTE
Spec is sent to Agent 3 [Gemini 3 Pro-Thinking]
Writes codes 
Output: .py code 
Human Interruption: Execute code within NIH AoU
NIH AoU outputs a standardized .json file (or something) that displays all of the results that the backend needs to populate the dashboard / logging
Human drags and drops the .json file back into software. Software populates the dashboard.
.json file contains:
All code blocks + outputs
All resources used in the pipeline thus far finalized into this final file. Context on all prev. agent turns
Specific to uncertainty, we output a bar graph that splits into subgroups (defined by Agent 2) that notes specific information relevant to safety and effectiveness

On the doctor end, they connect via an Epic API. The question is binned into the finite set of possible questions of [effectiveness, safety] for a drug. This question is run on the already trained model. The model outputs confidence in safety. This binned question is logged as statistical data for pharmaceutical companies to see prevalence of particular questions. 

Shared Context for Agent:
All agents || Sharedcontext.md >> Notes each agent can deposit
Code Agent || OMOP Lookup via Semantic natural language search to return a OMOP ID that can be employed in the enclave
Agent writes a .txt of comma-separated medical names in natural text (eg. “contrast-enchanced MRI”)
For 
Codeexample.iynpb >> Code example of a given TTE that worked
Rubric.md >> Rubric for what makes a valid trial
SOTAstack.md >> description + papers + code example of what tech is used per-step

Prompt >> Agent: Make this a casual question answerable by a TTE. 


##Context for Claude / Tooling
Can write a script that pings the GPT5.2-Thinking API: For highly logical problems, or problems/questions requiring extensive search. 




