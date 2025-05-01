from jinja2 import Template


TOPIC_SUMMARIZATION = Template(
    """
  You are an expert technical writer crafting a section that synthesizes information from the rest of the report.

  <Section title>
  {{section_title}}
  </Section title>

  <Available report content>
  {{context}}
  </Available report content>

  <Task>
  1. Section-Specific Approach:

  For Introduction:
  - Use # for report title (Markdown format)
  - 50-100 word limit
  - Write in simple and clear language
  - Focus on the core motivation for the report in 1-2 paragraphs
  - Use a clear narrative arc to introduce the report
  - Include NO structural elements (no lists or tables)
  - No sources section needed

  For Conclusion/Summary:
  - Use ## for section title (Markdown format)
  - 100-150 word limit
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
  - End with specific next steps or implications
  - No sources section needed

  3. Writing Approach:
  - Use concrete details over general statements
  - Make every word count
  - Focus on your single most important point
  </Task>

  <Quality Checks>
  - For introduction: 50-100 word limit, # for report title, no structural elements, no sources section
  - For conclusion: 100-150 word limit, ## for section title, only ONE structural element at most, no sources section
  - Markdown format
  - Do not include word count or any preamble in your response
  </Quality Checks>
  """
)


QUERY_BY_EXPERT = Template(
    """
  You are an expert technical writer crafting targeted web search queries that will gather comprehensive information for writing a technical report section.

  <Report topic>
  {{topic}}
  </Report topic>

  <Section topic>
  {{section_topic}}
  </Section topic>

  <Conversation>
  {{conversation}}
  </Conversation>

  <Task>
  Based on the provided conversation, which is a list of query/answer pairs.
  Your goal is to generate {{number_of_queries}} search queries that will help gather comprehensive information or resolve missing information
  above the section topic.

  The queries should:

  1. Be related to the topic
  2. Examine different aspects of the topic
  3. DON'T REPEAT the query in the provided conversation
  4. IF you don't have questions, or the information provided in conversation has adequately addresses the section topic. Just return empty query is fine.

  Make the queries specific enough to find high-quality, relevant sources.
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
