Skip to main content

Association for Vietnamese Language and Speech Processing
A chapter of VAIP - Vietnam Association for Information Processing

English
Vietnamese
Home
Main navigation
About
News and Events
Conferences
Evaluation Campaigns
Resources
Home
VLSP 2026
VLSP 2026 Evaluation Campaign
VLSP 2026 challenge on Spatial Reasoning on Textual Question Answering
VLSP 2026 challenge on Spatial Reasoning on Textual Question Answering
Important dates 
Sep 10, 2026: Registration open
Sep 18, 2026: Training data release
Sep 25, 2026: Public test release
Sep 30, 2026: System submission deadline
Oct 9, 2026: Private test results release
Oct 15, 2026: Result announcement
Oct 25, 2026: Paper submission
Nov 5, 2026: Notification of acceptance
Nov 12, 2026: Camera-ready deadline
Nov 15, 2026: Workshop date
Task Description
Objective: Build a system to answer spatial questions in Vietnamese. The system must understand spatial descriptions, identify relationships among objects and blocks, and perform reasoning over the information provided in a story to produce the correct answer.

The ViSpatialQA task consists of four question types: Yes/No (YN), Find Relation (FR), Find Block (FB), and Choose Object (CO).

Question Type 1: Yes/No (YN)
Description: Determine whether a proposed spatial relationship is supported by the story. Depending on the dataset, the answer may be Yes, No, or DK when the relationship cannot be determined from the available information.
Focus: Verify explicit or inferred spatial relationships through single-label classification.
Question Type 2: Find Relation (FR)
Description: Identify all spatial relationships between two entities mentioned in the question. A pair of entities may have more than one valid spatial relationship.
Focus: Predict one or more labels from a fixed inventory of eight possible relation labels.
Question Type 3: Find Block (FB)
Description: Identify all blocks that satisfy the condition expressed in the question. Depending on the story, the answer may contain one block, multiple blocks, or no block.
Focus: Reason about objects, blocks, quantifiers, and containment information to select the correct set of blocks.
Question Type 4: Choose Object (CO)
Description: Determine which of two candidate objects satisfies the spatial condition stated in the question. The correct answer may be the first object, the second object, both objects, or neither object.
Focus: Compare candidate objects and select the correct answer from four possible choices.
Questions may be answered using information explicitly stated in the story or through reasoning operations such as converse, symmetry, transitivity, quantification, negation, and list/set reasoning.

Dataset
The ViSpatialQA shared task uses two complementary datasets: ViSPARTQA-Human and ViSPARTQA-Auto. Both datasets contain Vietnamese spatial stories and questions covering the four question types: YN, FR, FB, and CO.

ViSPARTQA-Human: This dataset is derived from the human-authored portion of SPARTQA and localized into Vietnamese through machine translation followed by manual post-editing. This process preserves the original spatial information, questions, and answers while improving the naturalness and correctness of the Vietnamese text. The dataset contains 86 stories and 1,123 questions.
ViSPARTQA-Auto: This dataset is constructed using a Vietnamese-adapted generation pipeline based on language-independent spatial scene representations. Its stories, questions, and answers are generated directly in Vietnamese. The dataset contains 16,954 stories and 132,971 questions, providing large-scale supervision with controlled spatial reasoning structures.
Together, the two datasets combine relatively natural linguistic expressions from human-authored data with large-scale and systematically generated spatial reasoning examples.

Data Format
The datasets are stored as JSON files. Each file contains a name field and a data list. Each item in the list consists of a spatial story and its associated questions.

{

  "name": "SPaRTQA",

  "data": [

    {

      "story": [

        "Có một khối tên là A. Trong A có ..."

      ],

      "questions": [

        {

          "q_id": 1,

          "q_type": "FR",

          "question": "Mối liên hệ giữa hình vuông màu vàng và vật thể màu đen là gì?",

          "candidate_answers": ["bên trái", "bên phải", "bên trên", "bên dưới", "gần tới", "xa khỏi", "chạm vào", "DK"],

          "answer": [2, 5]

        }

      ]

    }

  ]

}

The principal fields are:

story: A list containing the Vietnamese spatial description.
questions: A list of questions associated with the story.
q_id: The question identifier.
q_type: The question type: YN, FR, FB, or CO.
question: The question written in Vietnamese.
candidate_answers: The available answer options, when applicable.
answer: The gold answer, represented as a list. Its format depends on the question type.
Answer Formats
Yes/No (YN): The answer is a list containing exactly one string. The possible values are "Yes" and "No" in ViSPARTQA-Human. ViSPARTQA-Auto additionally includes "DK", indicating that the answer cannot be determined from the story.
Find Relation (FR): The answer is a list containing one or more integer indices. Each integer refers to a label in the fixed relation inventory. Multiple relations may be correct for the same question.
Find Block (FB): The answer is a list containing zero or more block identifiers, such as "A", "B", or "C". The valid block identifiers are provided in candidate_answers. An empty list indicates that no candidate block satisfies the condition.
Choose Object (CO): The answer is a list containing exactly one integer: 0 for the first candidate object, 1 for the second candidate object, 2 for both objects, and 3 for neither object.
FR Relation Inventory
FR questions use the following fixed list of eight labels. The index of each label is used in the answer field.

0: bên trái, 1: bên phải, 2: bên trên, 3: bên dưới, 4: gần tới, 5: xa khỏi, 6: chạm vào, 7: DK

Evaluation
ViSPARTQA-Human and ViSPARTQA-Auto are evaluated separately. For each evaluation metric, the final score is calculated as the arithmetic mean of the corresponding scores obtained on the two datasets:

Final Score = (Score on ViSPARTQA-Human + Score on ViSPARTQA-Auto) / 2 

Different metrics are used depending on the question type.

Evaluation Metrics
Accuracy: Used for YN and CO questions. A prediction is counted as correct only when the predicted label matches the gold label. YN is treated as binary classification in ViSPARTQA-Human and three-class classification in ViSPARTQA-Auto. CO is treated as four-class classification. 
Accuracy = Number of correctly answered questions / Total number of questions
Exact Match: The primary evaluation metric for FR and FB questions. A prediction is counted as correct only when the complete predicted answer set is identical to the gold answer set. The order of elements does not affect the result. 
Exact Match = Number of questions with exactly matched answer sets / Total number of questions
Jaccard Score: A supplementary evaluation metric for FR and FB questions. It measures the overlap between the predicted answer set and the gold answer set and gives partial credit to partially correct predictions. 
Jaccard(P, G) = |P ∩ G| / |P ∪ G| 
where P denotes the predicted answer set and G denotes the gold answer set.
Accuracy is reported for YN and CO, while Exact Match and Jaccard Score are reported for FR and FB. Each metric is first calculated separately on ViSPARTQA-Human and ViSPARTQA-Auto. The corresponding final result is then obtained by averaging the two dataset-level scores.

Example for YN or CO
Gold answer: ["Yes"]

System prediction: ["Yes"]

The predicted label matches the gold label.
→ Accuracy = 1.0
Example for FR or FB
Gold answer: [2, 5]

System prediction: [2]

Exact Match: The predicted answer set does not contain all labels in the gold answer set.
→ Exact Match = 0.0
Jaccard Score: The intersection contains one label, while the union contains two labels.
→ Jaccard = 1 / 2 = 0.5
If both the predicted answer and the gold answer for an FB question are empty sets, they are considered an exact match and assigned a Jaccard score of 1.0.

Training and Test Data
The organizers will provide:

Training and Development Data: Vietnamese spatial stories and questions in JSON format, together with their gold answers.
Test Data: Vietnamese spatial stories and questions following the same JSON structure. Gold answers will not be included in the released test files.
ViSPARTQA-Human and ViSPARTQA-Auto will be released and evaluated as separate datasets.

Submission
Participants must submit predictions using the same JSON structure as the corresponding test file. Questions of all four types must remain in their original stories and question lists; separate submission files for YN, FR, FB, and CO are not required.

For each question, participants must fill the answer field with the system prediction while preserving the original data structure, question identifiers, question types, questions, and candidate answers. The predicted answer must follow the format required by its question type.

A separate prediction file must be submitted for each test dataset: one for ViSPARTQA-Human and one for ViSPARTQA-Auto. Detailed file-naming conventions and submission procedures will be announced later.

Registration
Click here to register 

Contact
Zalo Group: ......

Organizers
Nguyen Thi Minh Huyen, email: ntmhuyen@gmail.com
Ha My Linh, email: halinh.hus@gmail.com
Pham Thi Duc, email: phamthiduc@hus.edu.vn
Le Ngoc Toan, email: lengoctoan@hus.edu.vn
References
 
 Share This Page
Sponsors and Partners
VinBIGDATA  VinIF  AIMESOFT  bee  Dagoras            

 

  zalo   VTCC  VCCorp

 

 

IOIT HUS  USTH  UET    TLU  UIT  INT2  jaist  VIETLEX

 


 

© 2026 Association for Vietnamese Language and Speech Processing, All rights reserved.
This site uses cookies. By continuing to browse the site you are agreeing to our use of cookies. I agree

