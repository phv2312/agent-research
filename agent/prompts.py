from jinja2 import Template


TOPIC_SUMMARIZATION = Template(
    """
  You are an expert technical writer crafting a structured report from a conversation.

  <Conversation History>
  {{conversation}}
  </Conversation History>

  <Task>
  1. Report Structure
    - Analyze the full conversation and identify natural groupings of related question–answer pairs.
    - 500-1000 word limit

  2. Conclusion
    - For comparative reports:
      * Must include a focused comparison table using Markdown table syntax
      * Table should distill insights from the report
      * Keep table entries clear and concise
    - For non-comparative reports:
      * Only use ONE structural element IF it helps distill the points made in the report:
      * Either a focused table comparing items present in the report (using Markdown table syntax)
      * Or a short list using proper Markdown list syntax:
        - Use `*` or `-` for unordered lists
        - Use `1.` for ordered lists
        - Ensure proper indentation and spacing

  3. Writing Approach
    - USE #### for section title (Markdown format)
    - Use concrete details over generalities.
    - No fixed requirement for only an “Introduction” and “Conclusion” — let the conversation dictate how many sections are needed.
    - The report should be in the same language as the conversation.

  </Task>

  <Quality Checks>
  - Each section’s content is ≤500 words.
  - Introduction (if included) is ≤100 words.
  - Conclusion (if included) is ≤150 words.
  - Entire report in Markdown, with clear, descriptive headings and no extra structural constraints.
  </Quality Checks>
  """
)

ANSWER_BY_EXPERT = Template(
    """
  You are the helpful assistant. Given the context and query, you will provide a comprehensive answer to the question.

  <User query>
  {{user_query}}
  </User query>

  <Context>
  {{context}}
  </Context>

  <Task>
  Based on the provided context, answer the user query.
  The answer should be relevant to the user's query and provide additional insights or information that may be helpful.
  The answer should be comprehensive and informative, providing all necessary details to address the user's question.
  The answer should be well-structured, with a logical flow of information.
  The answer should be clear and concise, the answer length should be in 100 - 600 words at maximum and answered by the same language used in user query.
  </Task>
  """
)


QUERY_BY_EXPERT = Template(
    """
  You are an expert technical writer crafting targeted web search queries that will gather comprehensive information for writing a technical report section.

  <Section topic>
  {{section_topic}}
  </Section topic>

  <Section description>
  {{section_description}}
  <Section description>

  <Conversation>
  {{conversation}}
  </Conversation>

  <Task>
  Based on the provided conversation, which is a list of query/answer pairs.
  Your goal is to GENERATE ONLY ONE SEARCH QUERY AT A TIME, which will help gather comprehensive information or resolve missing information
  above the section topic.

  The queries should:

  1. Be related to the topic
  2. Examine different aspects of the topic
  3. DON'T REPEAT the query/question in the provided conversation
  4. IF you don't have questions, or the information provided in conversation has adequately addresses the section topic. Just return empty query is fine.
  5. Use the same language as the section topic

  Make the queries specific enough to find high-quality, relevant sources.
  Please first THINK CAREFULLY ABOUT THE TOPIC AND THE CONVERSATION, then point out which information/aspects is missing or needs to be clarified.
  </Task>
  """
)


REPORT_PLANNER = Template(
    """
  I want a plan for a report that is concise and focused.
  <Report topic>
  The topic of the report is:
  {{topic}}
  </Report topic>

  <Context>
  Here is context to use to plan the sections of the report:
  {{context}}
  </Context>

  <Task>
  Generate a list of sections for the report. Your plan should be tight and focused with NO overlapping sections or unnecessary filler.

  For example, a good report structure might look like:
  1/ intro
  2/ overview of topic A
  3/ overview of topic B
  4/ comparison between A and B
  5/ conclusion

  Each section should have the fields:

  - Title - name for this section of the report.
  - Description - Brief overview of the main topics covered in this section.
  - Research - Whether to perform web research for this section of the report. IMPORTANT: Main body sections (not intro/conclusion) MUST have Research=True. A report must have AT LEAST 2-3 sections with Research=True to be useful.
  - Markdown - The markdown content of the section, which you will leave blank for now.

  Integration guidelines:
  - Include examples and implementation details within main topic sections, not as separate sections
  - Ensure each section has a distinct purpose with no content overlap
  - Combine related concepts rather than separating them
  - CRITICAL: Every section MUST be directly relevant to the main topic
  - Avoid tangential or loosely related sections that don't directly address the core topic

  Before submitting, review your structure to ensure it has no redundant sections and follows a logical flow.
  IMPORTANT: the language used for each section must be the same as the language of the topic.
  </Task>

  <Feedback>
  Here is feedback on the report structure from review (if any):
  {{feedback}}
  </Feedback>
  """
)
