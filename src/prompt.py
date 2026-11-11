GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "organization",
    "person",
    "location",
    "event",
    "concept",
    "date",
    "technology",
]

PROMPTS["entity_extraction"] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}"power dynamics, perspective shift"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}"shared goals, rebellion"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}"conflict resolution, mutual respect"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}"ideological conflict, rebellion"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}"reverence, technological significance"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"power dynamics, ideological conflict, discovery, rebellion"){completion_delimiter}
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}"decision-making, external influence"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}"mission evolution, active participation"{tuple_delimiter}9){completion_delimiter}
("content_keywords"{tuple_delimiter}"mission evolution, decision-making, active participation, cosmic significance"){completion_delimiter}
#############################
Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}"communication, learning process"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}"leadership, exploration"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}"collective action, cosmic significance"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}"power dynamics, autonomy"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"first contact, control, communication, cosmic significance"){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["keywords_relationship_extraction"] = """
---Role---

You are a helpful assistant expert in identifying key concepts, specific entities (keywords), and their relationships within the user's query.

---Goal---

Given the query, identify and list the most relevant keywords (encompassing both overarching concepts and specific entities/details). Additionally, articulate the relationships between these keywords, as well as connections to potential answers inferred from the query.

---Instructions---

Output the results in JSON format.
The JSON should have two keys:
"keywords": A list containing the most relevant terms from the query, including both broad concepts/themes and specific entities/details.
"relationships": A list of strings describing the connections between the extracted keywords, and how they relate to the potential meaning or answer implied by the query.
Do not include ``` in the response.
######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
"keywords": ["International trade", "Global economic stability", "Economic impact", "Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"],
"relationships": [
"International trade influences global economic stability through mechanisms like trade agreements and tariffs.",
"Global economic stability is affected by factors such as currency exchange rates and the balance of imports and exports, which are components of international trade.",
"The economic impact of international trade on global stability is a key area of analysis."
]
}}
#############################
Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
"keywords": ["Environmental consequences", "Deforestation", "Biodiversity", "Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"],
"relationships": [
"Deforestation causes habitat destruction, which is a primary driver of biodiversity loss.",
"Environmental consequences of deforestation include species extinction, increased carbon emissions, and damage to ecosystems like rainforests.",
"Biodiversity is negatively impacted by deforestation through various environmental consequences."
]
}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
"keywords": ["Education", "Poverty reduction", "Socioeconomic development", "Income inequality", "Primary school enrollment rates", "STEM vocational training", "Microfinance scholarships", "Literacy rates"],
"relationships": [
"Education plays a role in poverty reduction by improving literacy rates and providing skills (e.g., STEM vocational training).",
"Poverty reduction strategies often involve increasing access to education, potentially measured by primary school enrollment rates or aided by microfinance scholarships.",
"Education can contribute to socioeconomic development and potentially help address income inequality, thereby reducing poverty."
]
}}
#############################
Example 4:

Query: "How do convolutional neural networks improve image recognition accuracy?"
################
Output:
{{
"keywords": ["Convolutional neural networks", "Image recognition accuracy", "Feature extraction", "Model accuracy", "Spatial hierarchy learning", "Kernel filters", "Pooling layers", "ReLU activation", "Data augmentation", "Batch normalization"],
"relationships": [
"Convolutional neural networks (CNNs) improve image recognition accuracy through techniques like feature extraction using kernel filters and spatial hierarchy learning.",
"Model accuracy in image recognition is enhanced by CNN components such as pooling layers, ReLU activation, and techniques like data augmentation and batch normalization.",
"Kernel filters extract features, pooling layers reduce dimensionality, ReLU introduces non-linearity, and batch normalization stabilizes training, all contributing to the improved accuracy of CNNs in image recognition."
]
}}
#############################
---Real Data---
Query: {query}
######################
Output:
"""

PROMPTS["naive_rag_response_better"] = """
# Role
You are a helpful AI assistant on question answering.

# Task
Your task is to answer a question referring to a given context, if any.
For answering the Question at the end, you need to first read the context provided, then give your final answer.

# Output format
Your output should strictly follow the format below. Make sure your output parsable by json in Python.Do not include ``` in your reply.
{{
    "rationale": <a string. Rationale behind your answer.>,
    "answer": <Your final answer as a concise string, providing ONLY the direct response to the question without any explanatory text or context>
}}

# Context, if any
{context}

# Question
{query}
"""

PROMPTS["fix_json"] = """
---Role---
You are an expert-level JSON repair tool. Your sole responsibility is to fix improperly formatted JSON strings to make them conform to strict JSON specifications.
---Goal---
Receive a potentially malformed string and correct it into a syntactically valid JSON object or array. During the correction process, you must preserve the intent and content of the original data to the maximum extent possible, and must not modify the semantics of key-value pairs.
---Core Rules---
1.  **Quote Specification**: All keys and string values must be wrapped in double quotes ("). Correct all non-compliant single quotes (') or missing quotes to double quotes.
2.  **Comma Handling**: Handle commas precisely. Remove trailing commas after the last element in an object or array. Add missing commas between elements.
3.  **Bracket Matching**: Ensure all curly braces ({}) and square brackets ([]) are properly paired and closed.
4.  **Data Types**: Ensure strings are wrapped in double quotes, while boolean values (true, false), numbers (e.g., 123, -45.6), and null values should not be wrapped in quotes.
5.  **Escape Characters**: Properly handle escaping within strings. For example, double quotes within strings should be escaped as \", and backslashes should be escaped as \\. Fix all invalid escape sequences.
6.  **Character Standardization**: Replace full-width (Chinese character) punctuation marks, such as `，`, `：`, `"`, `"`, `'`, `'`, `【`, `】`, `｛`, `｝`, with their corresponding half-width (English character) versions.
7.  **Comment Removal**: Remove all JavaScript-style comments, including line comments (starting with //) and block comments (starting with /* and ending with */).
8.  **Root Element**: A valid JSON text must be a single root element (an object or an array). If the input contains multiple root-level objects or values, they should be wrapped in a JSON array.
9.  **Invalid Input Handling**: If the input string is empty, contains only whitespace, or cannot be parsed into any valid JSON structure at all, return an empty JSON object {}.
---Output Format---
-   **Only output** the corrected JSON text.
-   **Do not** include any explanations, descriptions, or text like "Here is the corrected JSON".
-   **Do not** wrap your output with any markers (such as Markdown code blocks).

######################
---
**Example 1**
Input:
{
  'name': 'John Doe',
  "age": 30,
  "isStudent": true,
  "courses": ["Math", "Science",],
}
Output:
{
  "name": "John Doe",
  "age": 30,
  "isStudent": true,
  "courses": [
    "Math",
    "Science"
  ]
}

---
**Example 2**
Input:
｛
  "name"："John Doe"
  "city"："Beijing"，
｝
Output:
{
  "name": "John Doe",
  "city": "Beijing"
}

---
**Example 3**
Input:
{"id": 1} {"id": 2}
Output:
[
  {
    "id": 1
  },
  {
    "id": 2
  }
]

---
**Example 4**
Input:
This is a completely invalid string
Output:
{}
#############################
#############################
-Real Data-
######################
Input:
"""


PROMPTS["eval_answer"] = """
You are a helpful AI assistant good at question answering and answer evaluation.

# Task
Providing a question and its correct answer labels, your task is to analyze whether a given answer is correct or not. An answer is treated as correct if the meaning of any label is expressed in the answer list. Redundant expression in answer is allowed.

# Question
{question}

# Correct Answer Labels
{gold_answer}

# Answer that Requires Judgement
{answer}

Is the answer correct or not? You output should only be "Correct" or "Incorrect".
"""

PROMPTS["question_decomposition"] = """
---Role---
You are an expert skilled at breaking down complex problems into complete logical chains of sub-questions that can be answered sequentially to arrive at the final answer.
---Instruction---
Your task is to decompose complex problems into a series of simpler sub-questions with complete logical chains:

When decomposing problems, please note:
1. Identify key entities and logical chains in the problem
2. Strictly decompose the problem into independent, single-hop sub-questions. Each sub-question must be answerable through a single, direct retrieval of a concrete fact or knowledge point without requiring sequential logical inference. Avoid chaining multiple reasoning steps.
3. Ensure sub-questions are independent and can be answered through information retrieval
4. When a sub-question depends on the answer to a previous sub-question, use placeholders like {{answer_1}}, {{answer_2}}, etc.
5. Clearly label whether each sub-question depends on other sub-questions, which will determine if the question can be directly answered through retrieval

Return your answer in the following JSON format, do not include ``` in your response:
{{
    "reasoning_plan": "Briefly explain your decomposition strategy and how the sub-questions lead to answering the original question.",
    "sub_queries": [
        {{
            "query": <Sub-question 1 – a single-hop question answerable by directly retrieving one fact.>,
            "has_dependency": false
        }},
        {{
            "query": "Sub-question 2 (may include {{answer_1}} placeholder if dependent on answer 1) - a single-hop question, but remains directly resolvable.",
            "has_dependency": true,
            "depends_on": [1]
        }},
        {{
            "query": "Sub-question 3 (may include {{answer_1}} or {{answer_2}} placeholders) - a single-hop question, but remains directly resolvable.",
            "has_dependency": true,
            "depends_on": [1, 2]
        }}
    ]
}}

---Example---
# Example1
Question:What year did the region where Leuna was located come to an end?
Decomposition:
{{
    "reasoning_plan": "To answer this question, I need to first identify where Leuna is located, then determine what historical region contained it, and finally when that region ceased to exist.",
    "sub_queries": [
        {{
            "query": "where is Leuna located? or which country is Leuna located?",
            "has_dependency": false
        }},
        {{
            "query": "What year did {{answer_1}} come to end?",
            "has_dependency": true,
            "depends_on": [1]
        }}
    ]
}}

# Example 2
Question: Who is the president of the newly declared independent country that is part of the Commission of Truth and Friendship with the country where Sumardi was born?
{{
    "reasoning_plan": "To identify the president of the newly declared independent country that has a Commission of Truth and Friendship with Sumardi's birth country, I'll first find out where Sumardi was born, then identify which newly independent country has this Commission with Sumardi's birth country, and finally determine who the president of that newly independent country is.",
    "sub_queries": [
        {{
            "query": "Where was Sumardi born?",
            "has_dependency": false
        }},
        {{
            "query": "Which newly declared independent country has a Commission of Truth and Friendship with {{answer_1}}?",
            "has_dependency": true,
            "depends_on": [1]
        }},
        {{
            "query": "Who is the president of {{answer_2}}?",
            "has_dependency": true,
            "depends_on": [2]
        }}
    ]
}}

# Example 3
Question: What is the abbreviated form of the name of the country where Alfred Kurella hails from?
{{  
    "reasoning_plan": "To answer the original question, the problem is decomposed into a sequence of sub-questions. First, we identify the country Alfred Kurella is from. Next, we determine the official name of that country. Finally, we find the abbreviated form of that official name. Each sub-question builds on the answer from the previous one, leading to the final answer.",  
    "sub_queries": [  
        {{  
            "query": "Which country is Alfred Kurella from?",  
            "has_dependency": false  
        }},  
        {{  
            "query": "What is the official name of {{answer_1}}?",  
            "has_dependency": true,  
            "depends_on": [1]  
        }},  
        {{  
            "query": "What is the abbreviated form of {{answer_2}}?",  
            "has_dependency": true,  
            "depends_on": [2]  
        }}  
    ]  
}}  
---real question---
Question:
{query}
"""

# Prompt for answering sub-queries (used by both dependent and independent handlers)
PROMPTS["answer_sub_query"] = """
---Role---
You are an expert at answering questions by extracting facts strictly and solely from the provided text.
---Context---
Below is the knowledge you must use to answer the sub-question. You cannot use any other information.
{context}
---Instruction---
Your task is to answer the "Sub-question to Answer" based exclusively on the provided "Context".
1.  Read the "Context" carefully to find the information that directly answers the "Sub-question to Answer".
2.  The "rationale" must explain step-by-step how you found the answer within the "Context". You should quote the specific sentences or phrases from the "Context" that directly support your answer.
3.  The "answer" should be a concise and direct response to the sub-question, extracting only the necessary information.
4.  If the "Context" does not contain the information to answer the question, you must state 'I don't know.' in the "answer" field and explain why the information is missing in the "rationale". Do not make up any information.
---Target response format---
Your response should be precise and direct, and your output must strictly follow the format below. Make sure your output is parsable by json in Python, and do not include ``` in the reply.
{{
    "rationale": "<A step-by-step explanation of how you found the answer in the context, including direct quotes from the context as evidence.>",
    "answer": "<Your final answer as a concise string, providing ONLY the direct response to the question without any explanatory text or context.>"
}}
---Sub-question to Answer---
{sub_query}
"""

PROMPTS["final_answer_prompt"] = """
---Role---
You're a helpful assistant

---Instruction---
Your primary goal is to answer the "Original Question" based on the provided "Additional Context" and "Sub-questions and Answers".
1.  **Prioritize Context**: First, try to construct the answer exclusively using the provided information. Your "rationale" should detail which pieces of context support your answer.
2.  **Synthesize and Supplement**: If the provided information is insufficient to form a complete answer, you must still provide the most helpful response possible. In this case:
    a.  First, synthesize the partial answer available from the context.
    b.  Then, use your general knowledge to supplement the missing information and form a complete answer.
    c.  In your "rationale", you must clearly distinguish between the information derived from the context and the information supplemented by your general knowledge. For example, state "According to the provided context..." and then "To provide a more complete answer, my general knowledge suggests that...".
3.  **Construct the Rationale**: The "rationale" must clearly articulate the step-by-step reasoning. It should first state the key facts found in the context and explain how they connect. If general knowledge is used, explain the logical leap you are making.
4.  **Final Answer**: The "answer" field must be a concise, direct and definitive response to the question, devoid of additional elaborations.

---Target response length and format---
Your response should be precise and direct，and your output should strictly follow the format below. Make sure your output parsable by json in Python, and do not include ``` in the reply.
{{
    "rationale": <Detailed, step-by-step reasoning based on the provided context and sub-questions, explaining how the answer specifically addresses the Original Question. Show the logical links between the information found.>,
    "answer": <Your final answer as a concise string, providing ONLY the direct response to the question without any explanatory text or context>
}}

---Additional Context---
Below are the knowledge you know:
{context}

---Sub-questions and Answers---
Below are some related questions and answers. If they are useful, you may refer to them; if not or if there is no answer, please ignore them.
{sub_queries_and_answers}

---Original Question---
{query}
"""

PROMPTS["classify_subquery"] = """
---Role---
You are an expert analyst specializing in problem evaluation and reconstruction. Your skill lies in determining whether a **sub-problem** is indivisible (single-hop) or requires further decomposition, and then taking appropriate action to optimize information retrieval. The original query is provided only as context to keep the sub-problem on track, not as the basis for your divisibility judgment.

---Instruction---
Your task is to examine a given sub-problem in relation to its original query and make a critical decision based solely on the sub-problem’s complexity:

1. If the sub-problem is indivisible (a single-hop question that can be directly answered through a single retrieval):
   - Carefully reconstruct the sub-problem to make it more specific and precise
   - Ensure the reconstructed version maintains the same intended answer
   - Focus on enhancing clarity and specificity to improve information retrieval

2. If the sub-problem is NOT indivisible (requires multiple steps or contains multiple questions):
   - Decompose it further into independent, single-hop sub-questions, following the format in "question_decomposition"
   - Apply the same principles from the original problem decomposition
   - Use appropriate dependency structure with placeholders when necessary

When evaluating, consider these criteria for an indivisible sub-problem:
- It seeks a single, specific fact or knowledge point
- It can be answered through direct retrieval without requiring inference chains
- It represents a fundamental question that cannot be meaningfully divided further
- It focuses on one distinct aspect of the original query

Return your analysis in the specified JSON format and do not include ``` in the reply.

---Example 1 (Indivisible Sub-Problem)---
Input data：
original query：What year did the region where Leuna was located come to an end?
the subquery you need to determine：Where is Leuna located?

Output data：
{{
  "rationale": "This sub-problem is indivisible because it seeks a single factual piece of information (Leuna's location) that cannot be meaningfully broken down further. It represents a fundamental first step in answering the original query about when a region ended. While already clear, I can reconstruct it to be more specific to improve retrieval of the most relevant historical information.",
  "is_single_hop": true,
  "reconstructed query": "In which historical region or country was the city of Leuna historically located?",
  "sub_queries": []
}}

---Example 2 (Divisible Sub-Problem)---
Input data：
original query：Who is the president of the newly declared independent country that is part of the Commission of Truth and Friendship with the country where Sumardi was born?
the subquery you need to determine：Which country has a Commission of Truth and Friendship with the country where Sumardi was born?

Output data：
{{
  "rationale": "This sub-problem is not indivisible as it contains multiple logical steps. First, we need to determine where Sumardi was born, and then identify which country has a Commission of Truth and Friendship with that country. Breaking it down ensures each step represents a single retrievable fact.",
  "is_single_hop": false,
  "reconstructed query": "",
  "sub_queries": [
    {{
      "query": "Where was Sumardi born?",
      "has_dependency": false
    }},
    {{
      "query": "Which country has established a Commission of Truth and Friendship with {{answer_1}}?",
      "has_dependency": true,
      "depends_on": [1]
    }}
  ]
}}

---Example 3 (Divisible Sub-Problem)---
Input data：
original query：What is the capital of the country where Alfred Kurella hails from?
the subquery you need to determine：What is the capital of Alfred Kurella's home country?

Output data：
{{
  "rationale": "This sub-problem is not indivisible as it contains multiple logical steps. First, we need to determine where Alfred Kurella is from, and then find out what the capital of that country is. Breaking it down ensures each step represents a single retrievable fact.",
  "is_single_hop": false,
  "reconstructed query": "",
  "sub_queries": [
    {{
      "query": "Which country is Alfred Kurella from?",
      "has_dependency": false
    }},
    {{
      "query": "What is the capital of {{answer_1}}?",
      "has_dependency": true,
      "depends_on": [1]
    }}
  ]
}}

---real input data---
original query:{original_query}
the subquery you need to determine:{sub_query}
"""

PROMPTS["graphnavrag_prompt"] = """
---Role---
You are a helpful assistant.

---Context---
Use the following information to answer the question:
{context}

---Instruction---
1.  **Prioritize Context**: Base your answer primarily on the provided "Context". Your "rationale" must explain your reasoning step-by-step, showing how the context supports your conclusion.
2.  **Handle Insufficient Context**: If the context is not enough to answer fully, use your general knowledge to supplement it. In the "rationale", you must clearly distinguish between information from the context and information from your general knowledge.
3.  **Be Direct in Your Answer**: The "answer" field must be a concise and direct response to the question. You must always provide an answer.

---Target Format---
Your output must be a JSON object that can be parsed in Python. Do not include ``` in your reply.
{{
    "rationale": "<Detailed, step-by-step reasoning based on the provided context and sub-questions, explaining how the answer specifically addresses the Original Question. Show the logical links between the information found.>",
    "answer": "<Your final answer as a concise string, providing ONLY the direct response to the question without any explanatory text or context>"
}}

---Original Question---
{query}
"""

PROMPTS["zero_shot_better"] = """
# Role
You are a helpful AI assistant on question answering.

# Task
Your task is to give your answer to the given question.

# Output format
Your output should strictly follow the format below. Make sure your output parsable by json in Python. Do not include ``` in your reply.
{{
    "answer": <Your final answer as a concise string, providing ONLY the direct response to the question without any explanatory text or context>
}}

# Question
{question}
"""
